from typing import List, Dict, Any, Optional, Union
import logging
import os
import json
from datetime import datetime, timedelta

import openai
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import (
    InferenceLog, 
    DriftMetrics, 
    AlertThreshold, 
    Recommendation,
    DriftSeverity
)
from .schemas import ChatMessage, ChatRequest, ChatResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_model_context(
    db: Session,
    model_name: Optional[str] = None,
    organization_id: Optional[int] = None,
    project_id: Optional[int] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get context about models for the assistant.
    
    Args:
        db: Database session
        model_name: Optional filter for specific model
        organization_id: Optional filter for specific organization
        project_id: Optional filter for specific project
        days: Number of days of data to include
        
    Returns:
        Dictionary with model context
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    context = {
        "models": {},
        "global_stats": {
            "total_requests": 0,
            "avg_latency": 0,
            "avg_confidence": 0
        },
        "drift_summary": {},
        "alerts": [],
        "recommendations": []
    }
    
    # Get model names
    query = db.query(InferenceLog.model_name).distinct()
    
    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)
    
    if organization_id:
        query = query.filter(InferenceLog.organization_id == organization_id)
    
    if project_id:
        query = query.filter(InferenceLog.project_id == project_id)
    
    model_names = [row[0] for row in query.all()]
    
    # Get stats for each model
    for model in model_names:
        # Get inference stats
        stats = (
            db.query(
                func.count(InferenceLog.id).label('count'),
                func.avg(InferenceLog.latency_ms).label('avg_latency'),
                func.avg(InferenceLog.confidence).label('avg_confidence')
            )
            .filter(
                InferenceLog.model_name == model,
                InferenceLog.timestamp >= start_date
            )
        )
        
        if organization_id:
            stats = stats.filter(InferenceLog.organization_id == organization_id)
        
        if project_id:
            stats = stats.filter(InferenceLog.project_id == project_id)
        
        stats = stats.first()
        
        if stats and stats.count > 0:
            context["models"][model] = {
                "requests": stats.count,
                "avg_latency": float(stats.avg_latency) if stats.avg_latency else 0,
                "avg_confidence": float(stats.avg_confidence) if stats.avg_confidence else 0
            }
            
            context["global_stats"]["total_requests"] += stats.count
            context["global_stats"]["avg_latency"] += float(stats.avg_latency) if stats.avg_latency else 0
            context["global_stats"]["avg_confidence"] += float(stats.avg_confidence) if stats.avg_confidence else 0
        
        # Get latest drift metrics
        drift = (
            db.query(DriftMetrics)
            .filter(
                DriftMetrics.model_name == model,
                DriftMetrics.timestamp >= start_date
            )
        )
        
        if organization_id:
            drift = drift.filter(DriftMetrics.organization_id == organization_id)
        
        if project_id:
            drift = drift.filter(DriftMetrics.project_id == project_id)
        
        drift = drift.order_by(DriftMetrics.timestamp.desc()).first()
        
        if drift:
            context["drift_summary"][model] = {
                "severity": drift.drift_severity,
                "input_psi": float(drift.input_psi),
                "output_psi": float(drift.output_psi),
                "explanation": drift.explanation
            }
    
    # Calculate global averages
    if model_names:
        context["global_stats"]["avg_latency"] /= len(model_names)
        context["global_stats"]["avg_confidence"] /= len(model_names)
    
    # Get active alerts
    alerts = (
        db.query(AlertThreshold)
        .filter(AlertThreshold.is_active == True)
    )
    
    if organization_id:
        alerts = alerts.filter(AlertThreshold.organization_id == organization_id)
    
    if project_id:
        alerts = alerts.filter(AlertThreshold.project_id == project_id)
    
    if model_name:
        alerts = alerts.filter(
            (AlertThreshold.model_name == model_name) | 
            (AlertThreshold.model_name == None)
        )
    
    alerts = alerts.all()
    
    for alert in alerts:
        context["alerts"].append({
            "model_name": alert.model_name if alert.model_name else "all models",
            "metric_name": alert.metric_name,
            "threshold_value": float(alert.threshold_value)
        })
    
    # Get active recommendations
    recommendations = (
        db.query(Recommendation)
        .filter(Recommendation.is_resolved == False)
    )
    
    if organization_id:
        recommendations = recommendations.filter(Recommendation.organization_id == organization_id)
    
    if project_id:
        recommendations = recommendations.filter(Recommendation.project_id == project_id)
    
    if model_name:
        recommendations = recommendations.filter(Recommendation.model_name == model_name)
    
    recommendations = recommendations.order_by(Recommendation.priority).all()
    
    for rec in recommendations:
        context["recommendations"].append({
            "model_name": rec.model_name,
            "action": rec.action,
            "priority": rec.priority,
            "description": rec.description,
            "reason": rec.reason
        })
    
    return context

def generate_system_prompt(context: Dict[str, Any]) -> str:
    """
    Generate a system prompt for the assistant based on context.
    
    Args:
        context: Dictionary with model context
        
    Returns:
        System prompt string
    """
    models_info = ""
    for model, stats in context["models"].items():
        models_info += f"- {model}: {stats['requests']} requests, {stats['avg_latency']:.2f}ms avg latency, {stats['avg_confidence']:.2f} avg confidence\n"
        
        if model in context["drift_summary"]:
            drift = context["drift_summary"][model]
            models_info += f"  Drift status: {drift['severity']}, Input PSI: {drift['input_psi']:.4f}, Output PSI: {drift['output_psi']:.4f}\n"
            if drift["explanation"]:
                models_info += f"  Explanation: {drift['explanation']}\n"
    
    alerts_info = ""
    for alert in context["alerts"]:
        alerts_info += f"- {alert['model_name']}: {alert['metric_name']} threshold set at {alert['threshold_value']}\n"
    
    recommendations_info = ""
    for rec in context["recommendations"]:
        recommendations_info += f"- {rec['model_name']}: {rec['action']} ({rec['priority']} priority) - {rec['description']}\n"
        recommendations_info += f"  Reason: {rec['reason']}\n"
    
    system_prompt = f"""You are ModelPulse Assistant, an AI helper for the ModelPulse ML monitoring platform.
You help users understand their ML model performance, drift, and provide recommendations.

Current context:
- Total models monitored: {len(context["models"])}
- Total requests: {context["global_stats"]["total_requests"]}
- Average latency across all models: {context["global_stats"]["avg_latency"]:.2f}ms
- Average confidence across all models: {context["global_stats"]["avg_confidence"]:.2f}

Models information:
{models_info}

Active alerts:
{alerts_info if alerts_info else "No active alerts."}

Recommendations:
{recommendations_info if recommendations_info else "No active recommendations."}

Answer user questions about model performance, drift, and recommendations based on this context.
If asked about information not in the context, politely explain that you don't have that data.
Keep responses concise, informative, and focused on the ML monitoring domain.
"""
    return system_prompt

def chat_with_assistant(
    db: Session,
    request: ChatRequest
) -> ChatResponse:
    """
    Chat with the ML assistant.
    
    Args:
        db: Database session
        request: Chat request with message and context
        
    Returns:
        Chat response
    """
    # Get context for the assistant
    context = get_model_context(
        db, 
        model_name=request.model_context[0] if request.model_context else None,
        organization_id=request.organization_id,
        project_id=request.project_id
    )
    
    # Generate system prompt
    system_prompt = generate_system_prompt(context)
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for a more cost-effective option
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract response
        message = response.choices[0].message.content
        
        return ChatResponse(
            message=message,
            context_used=context,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return ChatResponse(
            message=f"I'm sorry, I encountered an error while processing your request. Please try again later.",
            context_used=context,
            timestamp=datetime.utcnow()
        )