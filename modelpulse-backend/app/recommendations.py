from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import (
    InferenceLog, 
    DriftMetrics, 
    AlertThreshold, 
    Recommendation, 
    RecommendationAction, 
    RecommendationPriority,
    DriftSeverity
)
from .schemas import RecommendationCreate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_drift_recommendations(
    db: Session,
    model_name: str,
    organization_id: int,
    project_id: Optional[int] = None,
    days_to_analyze: int = 7
) -> List[RecommendationCreate]:
    """
    Generate recommendations based on drift metrics.
    
    Args:
        db: Database session
        model_name: Name of the model to analyze
        organization_id: Organization ID
        project_id: Optional project ID
        days_to_analyze: Number of days of data to analyze
        
    Returns:
        List of recommendation objects
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days_to_analyze)
    
    # Get drift metrics for the specified period
    drift_metrics = (
        db.query(DriftMetrics)
        .filter(
            DriftMetrics.model_name == model_name,
            DriftMetrics.organization_id == organization_id,
            DriftMetrics.timestamp >= start_date
        )
    )
    
    if project_id:
        drift_metrics = drift_metrics.filter(DriftMetrics.project_id == project_id)
    
    drift_metrics = drift_metrics.order_by(DriftMetrics.timestamp.desc()).all()
    
    if not drift_metrics:
        logger.info(f"No drift metrics found for model {model_name}")
        return []
    
    recommendations = []
    
    # Check for persistent drift
    critical_count = 0
    warning_count = 0
    
    for metric in drift_metrics:
        if metric.drift_severity == DriftSeverity.CRITICAL:
            critical_count += 1
        elif metric.drift_severity == DriftSeverity.WARNING:
            warning_count += 1
    
    # If there are multiple critical drift events, recommend retraining
    if critical_count >= 3:
        recommendations.append(
            RecommendationCreate(
                model_name=model_name,
                action=RecommendationAction.RETRAIN,
                priority=RecommendationPriority.HIGH,
                description=f"Retrain model {model_name} due to persistent critical drift",
                reason=f"Detected {critical_count} critical drift events in the last {days_to_analyze} days",
                organization_id=organization_id,
                project_id=project_id
            )
        )
    # If there are multiple warning drift events, recommend monitoring
    elif warning_count >= 3:
        recommendations.append(
            RecommendationCreate(
                model_name=model_name,
                action=RecommendationAction.MONITOR,
                priority=RecommendationPriority.MEDIUM,
                description=f"Closely monitor model {model_name} for continued drift",
                reason=f"Detected {warning_count} warning drift events in the last {days_to_analyze} days",
                organization_id=organization_id,
                project_id=project_id
            )
        )
    
    # Check for sudden drift (latest drift is critical but previous ones were not)
    if drift_metrics and len(drift_metrics) > 1:
        latest = drift_metrics[0]
        previous = drift_metrics[1]
        
        if (latest.drift_severity == DriftSeverity.CRITICAL and 
            previous.drift_severity != DriftSeverity.CRITICAL):
            recommendations.append(
                RecommendationCreate(
                    model_name=model_name,
                    action=RecommendationAction.INVESTIGATE,
                    priority=RecommendationPriority.HIGH,
                    description=f"Investigate sudden drift in model {model_name}",
                    reason="Detected sudden change from normal to critical drift",
                    organization_id=organization_id,
                    project_id=project_id
                )
            )
    
    return recommendations

def generate_performance_recommendations(
    db: Session,
    model_name: str,
    organization_id: int,
    project_id: Optional[int] = None,
    days_to_analyze: int = 7
) -> List[RecommendationCreate]:
    """
    Generate recommendations based on model performance metrics.
    
    Args:
        db: Database session
        model_name: Name of the model to analyze
        organization_id: Organization ID
        project_id: Optional project ID
        days_to_analyze: Number of days of data to analyze
        
    Returns:
        List of recommendation objects
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days_to_analyze)
    
    # Get performance metrics for the specified period
    performance_data = (
        db.query(
            func.avg(InferenceLog.latency_ms).label('avg_latency'),
            func.avg(InferenceLog.confidence).label('avg_confidence'),
            func.count(InferenceLog.id).label('count')
        )
        .filter(
            InferenceLog.model_name == model_name,
            InferenceLog.organization_id == organization_id,
            InferenceLog.timestamp >= start_date
        )
    )
    
    if project_id:
        performance_data = performance_data.filter(InferenceLog.project_id == project_id)
    
    performance_data = performance_data.first()
    
    if not performance_data or performance_data.count == 0:
        logger.info(f"No performance data found for model {model_name}")
        return []
    
    recommendations = []
    
    # Check for high latency
    if performance_data.avg_latency and performance_data.avg_latency > 500:  # 500ms threshold
        recommendations.append(
            RecommendationCreate(
                model_name=model_name,
                action=RecommendationAction.OPTIMIZE,
                priority=RecommendationPriority.MEDIUM,
                description=f"Optimize model {model_name} for better latency",
                reason=f"Average latency of {performance_data.avg_latency:.2f}ms exceeds recommended threshold of 500ms",
                organization_id=organization_id,
                project_id=project_id
            )
        )
    
    # Check for low confidence
    if performance_data.avg_confidence and performance_data.avg_confidence < 0.7:  # 70% threshold
        recommendations.append(
            RecommendationCreate(
                model_name=model_name,
                action=RecommendationAction.RETRAIN,
                priority=RecommendationPriority.MEDIUM,
                description=f"Retrain model {model_name} to improve confidence",
                reason=f"Average confidence of {performance_data.avg_confidence:.2f} is below recommended threshold of 0.7",
                organization_id=organization_id,
                project_id=project_id
            )
        )
    
    # Check for performance degradation over time
    # Compare recent performance with older performance
    mid_date = now - timedelta(days=days_to_analyze/2)
    
    recent_performance = (
        db.query(
            func.avg(InferenceLog.latency_ms).label('avg_latency'),
            func.avg(InferenceLog.confidence).label('avg_confidence')
        )
        .filter(
            InferenceLog.model_name == model_name,
            InferenceLog.organization_id == organization_id,
            InferenceLog.timestamp >= mid_date
        )
    )
    
    older_performance = (
        db.query(
            func.avg(InferenceLog.latency_ms).label('avg_latency'),
            func.avg(InferenceLog.confidence).label('avg_confidence')
        )
        .filter(
            InferenceLog.model_name == model_name,
            InferenceLog.organization_id == organization_id,
            InferenceLog.timestamp >= start_date,
            InferenceLog.timestamp < mid_date
        )
    )
    
    if project_id:
        recent_performance = recent_performance.filter(InferenceLog.project_id == project_id)
        older_performance = older_performance.filter(InferenceLog.project_id == project_id)
    
    recent_performance = recent_performance.first()
    older_performance = older_performance.first()
    
    if recent_performance and older_performance:
        # Check for latency degradation (20% increase)
        if (recent_performance.avg_latency and older_performance.avg_latency and
            recent_performance.avg_latency > older_performance.avg_latency * 1.2):
            recommendations.append(
                RecommendationCreate(
                    model_name=model_name,
                    action=RecommendationAction.INVESTIGATE,
                    priority=RecommendationPriority.HIGH,
                    description=f"Investigate latency degradation in model {model_name}",
                    reason=f"Latency increased from {older_performance.avg_latency:.2f}ms to {recent_performance.avg_latency:.2f}ms",
                    organization_id=organization_id,
                    project_id=project_id
                )
            )
        
        # Check for confidence degradation (10% decrease)
        if (recent_performance.avg_confidence and older_performance.avg_confidence and
            recent_performance.avg_confidence < older_performance.avg_confidence * 0.9):
            recommendations.append(
                RecommendationCreate(
                    model_name=model_name,
                    action=RecommendationAction.RETRAIN,
                    priority=RecommendationPriority.HIGH,
                    description=f"Retrain model {model_name} due to confidence degradation",
                    reason=f"Confidence decreased from {older_performance.avg_confidence:.2f} to {recent_performance.avg_confidence:.2f}",
                    organization_id=organization_id,
                    project_id=project_id
                )
            )
    
    return recommendations

def get_recommendations(
    db: Session,
    model_name: Optional[str] = None,
    organization_id: Optional[int] = None,
    project_id: Optional[int] = None,
    include_resolved: bool = False,
    limit: int = 100
) -> List[Recommendation]:
    """
    Get recommendations from the database.
    
    Args:
        db: Database session
        model_name: Optional filter for specific model
        organization_id: Optional filter for specific organization
        project_id: Optional filter for specific project
        include_resolved: Whether to include resolved recommendations
        limit: Maximum number of recommendations to return
        
    Returns:
        List of recommendation objects
    """
    query = db.query(Recommendation)
    
    if model_name:
        query = query.filter(Recommendation.model_name == model_name)
    
    if organization_id:
        query = query.filter(Recommendation.organization_id == organization_id)
    
    if project_id:
        query = query.filter(Recommendation.project_id == project_id)
    
    if not include_resolved:
        query = query.filter(Recommendation.is_resolved == False)
    
    return query.order_by(Recommendation.priority, desc(Recommendation.created_at)).limit(limit).all()

def generate_all_recommendations(
    db: Session,
    organization_id: int,
    project_id: Optional[int] = None,
    days_to_analyze: int = 7
) -> List[Recommendation]:
    """
    Generate recommendations for all models in an organization/project.
    
    Args:
        db: Database session
        organization_id: Organization ID
        project_id: Optional project ID
        days_to_analyze: Number of days of data to analyze
        
    Returns:
        List of created recommendation objects
    """
    # Get all models for the organization/project
    query = (
        db.query(InferenceLog.model_name)
        .filter(InferenceLog.organization_id == organization_id)
        .distinct()
    )
    
    if project_id:
        query = query.filter(InferenceLog.project_id == project_id)
    
    model_names = [row[0] for row in query.all()]
    
    created_recommendations = []
    
    for model_name in model_names:
        # Generate drift recommendations
        drift_recs = generate_drift_recommendations(
            db, model_name, organization_id, project_id, days_to_analyze
        )
        
        # Generate performance recommendations
        perf_recs = generate_performance_recommendations(
            db, model_name, organization_id, project_id, days_to_analyze
        )
        
        # Combine recommendations
        all_recs = drift_recs + perf_recs
        
        # Save recommendations to database
        for rec in all_recs:
            # Check if a similar recommendation already exists
            existing = (
                db.query(Recommendation)
                .filter(
                    Recommendation.model_name == rec.model_name,
                    Recommendation.action == rec.action,
                    Recommendation.is_resolved == False,
                    Recommendation.created_at >= datetime.utcnow() - timedelta(days=1)
                )
            )
            
            if project_id:
                existing = existing.filter(Recommendation.project_id == project_id)
            
            existing = existing.first()
            
            if not existing:
                db_rec = Recommendation(
                    model_name=rec.model_name,
                    action=rec.action,
                    priority=rec.priority,
                    description=rec.description,
                    reason=rec.reason,
                    organization_id=rec.organization_id,
                    project_id=rec.project_id,
                    created_at=datetime.utcnow()
                )
                db.add(db_rec)
                db.commit()
                db.refresh(db_rec)
                created_recommendations.append(db_rec)
    
    return created_recommendations