"""
Automation module for ModelPulse.

This module contains functions for automating model retraining, validation, and deployment
based on drift detection, performance metrics, and user-defined policies.
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import os
import tempfile
import mlflow
from mlflow.tracking import MlflowClient

from .models import (
    AutomationPolicy,
    AutomationPolicyType,
    AutomationAction,
    RetrainingJob,
    ModelValidation,
    ModelDeployment,
    JobStatus,
    ModelValidationStatus,
    DriftMetrics,
    DriftSeverity,
    InferenceLog,
    Integration,
    IntegrationType,
)
from .drift import compute_drift_metrics
from .webhooks import (
    on_retraining_started,
    on_retraining_completed,
    on_model_deployed,
    on_validation_failed,
)


logger = logging.getLogger(__name__)


def check_policy_conditions(
    policy: AutomationPolicy, db: Session, current_time: Optional[datetime] = None
) -> bool:
    """
    Check if the conditions of an automation policy are met.
    
    Args:
        policy: The automation policy to check
        db: Database session
        current_time: Current time (for testing)
        
    Returns:
        True if conditions are met, False otherwise
    """
    if not current_time:
        current_time = datetime.utcnow()
        
    if not policy.is_active:
        return False
        
    if policy.policy_type == AutomationPolicyType.SCHEDULE:
        # Schedule-based policy
        schedule = policy.conditions.get("schedule", {})
        frequency = schedule.get("frequency")
        last_run = schedule.get("last_run")
        
        if not frequency or not last_run:
            return False
            
        last_run_time = datetime.fromisoformat(last_run)
        
        if frequency == "daily" and current_time - last_run_time >= timedelta(days=1):
            return True
        elif frequency == "weekly" and current_time - last_run_time >= timedelta(weeks=1):
            return True
        elif frequency == "monthly" and current_time - last_run_time >= timedelta(days=30):
            return True
        else:
            return False
            
    elif policy.policy_type == AutomationPolicyType.DRIFT:
        # Drift-based policy
        drift_threshold = policy.conditions.get("drift_severity")
        days_to_check = policy.conditions.get("days_to_check", 1)
        
        if not drift_threshold:
            return False
            
        # Get the latest drift metrics for the model
        query = db.query(DriftMetrics)
        
        if policy.model_name:
            query = query.filter(DriftMetrics.model_name == policy.model_name)
            
        query = query.filter(
            DriftMetrics.organization_id == policy.organization_id,
            DriftMetrics.timestamp >= current_time - timedelta(days=days_to_check)
        )
        
        if policy.project_id:
            query = query.filter(DriftMetrics.project_id == policy.project_id)
            
        latest_drift = query.order_by(desc(DriftMetrics.timestamp)).first()
        
        if not latest_drift:
            return False
            
        # Check if the drift severity meets the threshold
        severity_levels = {
            DriftSeverity.OK: 0,
            DriftSeverity.WARNING: 1,
            DriftSeverity.CRITICAL: 2,
        }
        
        threshold_level = severity_levels.get(drift_threshold, 0)
        current_level = severity_levels.get(latest_drift.drift_severity, 0)
        
        return current_level >= threshold_level
        
    elif policy.policy_type == AutomationPolicyType.PERFORMANCE:
        # Performance-based policy
        metric = policy.conditions.get("metric")
        threshold = policy.conditions.get("threshold")
        operator = policy.conditions.get("operator", "gt")  # gt, lt, eq
        days_to_check = policy.conditions.get("days_to_check", 1)
        
        if not metric or threshold is None:
            return False
            
        # Get the latest inference logs for the model
        query = db.query(InferenceLog)
        
        if policy.model_name:
            query = query.filter(InferenceLog.model_name == policy.model_name)
            
        query = query.filter(
            InferenceLog.organization_id == policy.organization_id,
            InferenceLog.timestamp >= current_time - timedelta(days=days_to_check)
        )
        
        if policy.project_id:
            query = query.filter(InferenceLog.project_id == policy.project_id)
            
        logs = query.all()
        
        if not logs:
            return False
            
        # Calculate the average metric value
        if metric == "latency_ms":
            values = [log.latency_ms for log in logs if log.latency_ms is not None]
        elif metric == "confidence":
            values = [log.confidence for log in logs if log.confidence is not None]
        else:
            return False
            
        if not values:
            return False
            
        avg_value = sum(values) / len(values)
        
        # Check if the metric meets the threshold
        if operator == "gt":
            return avg_value > threshold
        elif operator == "lt":
            return avg_value < threshold
        elif operator == "eq":
            return avg_value == threshold
        else:
            return False
            
    return False


def trigger_automation_policies(
    db: Session, organization_id: int, project_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Check all automation policies and trigger actions if conditions are met.
    
    Args:
        db: Database session
        organization_id: Organization ID
        project_id: Optional project ID
        
    Returns:
        List of triggered actions
    """
    triggered_actions = []
    
    # Get all active policies
    query = db.query(AutomationPolicy).filter(
        AutomationPolicy.organization_id == organization_id,
        AutomationPolicy.is_active == True
    )
    
    if project_id:
        query = query.filter(AutomationPolicy.project_id == project_id)
        
    policies = query.all()
    
    for policy in policies:
        if check_policy_conditions(policy, db):
            # Execute actions
            for action_config in policy.actions:
                action_type = action_config.get("type")
                
                if action_type == AutomationAction.RETRAIN:
                    job = trigger_retraining(
                        db=db,
                        model_name=policy.model_name,
                        policy_id=policy.id,
                        organization_id=organization_id,
                        project_id=project_id,
                        training_params=action_config.get("params", {})
                    )
                    
                    triggered_actions.append({
                        "policy_id": policy.id,
                        "action": "retrain",
                        "model_name": policy.model_name,
                        "job_id": job.id
                    })
                    
                elif action_type == AutomationAction.DEPLOY:
                    # For direct deployment without retraining
                    model_version = action_config.get("model_version")
                    integration_id = action_config.get("integration_id")
                    
                    if model_version and integration_id:
                        deployment = deploy_model(
                            db=db,
                            model_name=policy.model_name,
                            model_version=model_version,
                            integration_id=integration_id,
                            organization_id=organization_id,
                            project_id=project_id,
                            deployment_params=action_config.get("params", {})
                        )
                        
                        triggered_actions.append({
                            "policy_id": policy.id,
                            "action": "deploy",
                            "model_name": policy.model_name,
                            "model_version": model_version,
                            "deployment_id": deployment.id
                        })
                
                # Update last run time for schedule-based policies
                if policy.policy_type == AutomationPolicyType.SCHEDULE:
                    conditions = policy.conditions.copy()
                    if "schedule" in conditions:
                        schedule = conditions["schedule"].copy()
                        schedule["last_run"] = datetime.utcnow().isoformat()
                        conditions["schedule"] = schedule
                        policy.conditions = conditions
                        db.commit()
    
    return triggered_actions


def trigger_retraining(
    db: Session,
    model_name: str,
    organization_id: int,
    project_id: Optional[int] = None,
    policy_id: Optional[int] = None,
    integration_id: Optional[int] = None,
    training_params: Optional[Dict[str, Any]] = None
) -> RetrainingJob:
    """
    Create and start a retraining job.
    
    Args:
        db: Database session
        model_name: Name of the model to retrain
        organization_id: Organization ID
        project_id: Optional project ID
        policy_id: Optional policy ID that triggered the retraining
        integration_id: Optional integration ID to use for retraining
        training_params: Optional training parameters
        
    Returns:
        Created retraining job
    """
    # Create retraining job
    job = RetrainingJob(
        model_name=model_name,
        status=JobStatus.PENDING,
        training_params=training_params or {},
        policy_id=policy_id,
        integration_id=integration_id,
        organization_id=organization_id,
        project_id=project_id,
        started_at=datetime.utcnow()
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Trigger webhook
    on_retraining_started(db, job)
    
    # Start retraining in background (in a real implementation, this would be a Celery task)
    # For now, we'll just update the job status
    job.status = JobStatus.RUNNING
    db.commit()
    
    # In a real implementation, we would start the retraining process here
    # For now, we'll just simulate it by updating the job status
    
    return job


def complete_retraining_job(
    db: Session,
    job_id: int,
    status: JobStatus,
    metrics: Optional[Dict[str, Any]] = None,
    logs: Optional[str] = None
) -> RetrainingJob:
    """
    Complete a retraining job and trigger validation if needed.
    
    Args:
        db: Database session
        job_id: ID of the retraining job
        status: Final status of the job
        metrics: Optional training metrics
        logs: Optional training logs
        
    Returns:
        Updated retraining job
    """
    job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
    
    if not job:
        raise ValueError(f"Retraining job with ID {job_id} not found")
        
    job.status = status
    job.metrics = metrics or {}
    job.logs = logs
    job.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(job)
    
    # Trigger webhook
    on_retraining_completed(db, job)
    
    # If job was successful and there's a policy with validation action, trigger validation
    if status == JobStatus.COMPLETED and job.policy_id:
        policy = db.query(AutomationPolicy).filter(AutomationPolicy.id == job.policy_id).first()
        
        if policy:
            for action_config in policy.actions:
                action_type = action_config.get("type")
                
                if action_type == AutomationAction.VALIDATE:
                    # Get the model version from the training metrics
                    model_version = job.metrics.get("model_version")
                    
                    if model_version:
                        validation = validate_model(
                            db=db,
                            model_name=job.model_name,
                            model_version=model_version,
                            retraining_job_id=job.id,
                            organization_id=job.organization_id,
                            project_id=job.project_id,
                            validation_params=action_config.get("params", {})
                        )
                        
    return job


def validate_model(
    db: Session,
    model_name: str,
    model_version: str,
    organization_id: int,
    project_id: Optional[int] = None,
    retraining_job_id: Optional[int] = None,
    validation_params: Optional[Dict[str, Any]] = None
) -> ModelValidation:
    """
    Create and start a model validation.
    
    Args:
        db: Database session
        model_name: Name of the model to validate
        model_version: Version of the model to validate
        organization_id: Organization ID
        project_id: Optional project ID
        retraining_job_id: Optional retraining job ID
        validation_params: Optional validation parameters
        
    Returns:
        Created model validation
    """
    # Create model validation
    validation = ModelValidation(
        model_name=model_name,
        model_version=model_version,
        status=ModelValidationStatus.PENDING,
        validation_params=validation_params or {},
        retraining_job_id=retraining_job_id,
        organization_id=organization_id,
        project_id=project_id,
        started_at=datetime.utcnow()
    )
    
    db.add(validation)
    db.commit()
    db.refresh(validation)
    
    # Start validation in background (in a real implementation, this would be a Celery task)
    # For now, we'll just update the validation status
    validation.status = ModelValidationStatus.PENDING
    db.commit()
    
    # In a real implementation, we would start the validation process here
    # For now, we'll just simulate it
    
    return validation


def complete_model_validation(
    db: Session,
    validation_id: int,
    status: ModelValidationStatus,
    metrics: Optional[Dict[str, Any]] = None,
    logs: Optional[str] = None
) -> ModelValidation:
    """
    Complete a model validation and trigger deployment if needed.
    
    Args:
        db: Database session
        validation_id: ID of the model validation
        status: Final status of the validation
        metrics: Optional validation metrics
        logs: Optional validation logs
        
    Returns:
        Updated model validation
    """
    validation = db.query(ModelValidation).filter(ModelValidation.id == validation_id).first()
    
    if not validation:
        raise ValueError(f"Model validation with ID {validation_id} not found")
        
    validation.status = status
    validation.metrics = metrics or {}
    validation.logs = logs
    validation.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(validation)
    
    # If validation failed, trigger webhook
    if status == ModelValidationStatus.FAILED:
        on_validation_failed(db, validation)
        return validation
        
    # If validation was successful and there's a retraining job with a policy, check for deployment action
    if status == ModelValidationStatus.PASSED and validation.retraining_job_id:
        job = db.query(RetrainingJob).filter(RetrainingJob.id == validation.retraining_job_id).first()
        
        if job and job.policy_id:
            policy = db.query(AutomationPolicy).filter(AutomationPolicy.id == job.policy_id).first()
            
            if policy:
                for action_config in policy.actions:
                    action_type = action_config.get("type")
                    
                    if action_type == AutomationAction.DEPLOY:
                        integration_id = action_config.get("integration_id")
                        
                        if integration_id:
                            deployment = deploy_model(
                                db=db,
                                model_name=validation.model_name,
                                model_version=validation.model_version,
                                integration_id=integration_id,
                                organization_id=validation.organization_id,
                                project_id=validation.project_id,
                                retraining_job_id=validation.retraining_job_id,
                                validation_id=validation.id,
                                deployment_params=action_config.get("params", {})
                            )
                            
    return validation


def deploy_model(
    db: Session,
    model_name: str,
    model_version: str,
    integration_id: int,
    organization_id: int,
    project_id: Optional[int] = None,
    retraining_job_id: Optional[int] = None,
    validation_id: Optional[int] = None,
    deployment_params: Optional[Dict[str, Any]] = None,
    previous_deployment_id: Optional[int] = None,
    is_rollback: bool = False
) -> ModelDeployment:
    """
    Create and start a model deployment.
    
    Args:
        db: Database session
        model_name: Name of the model to deploy
        model_version: Version of the model to deploy
        integration_id: Integration ID to use for deployment
        organization_id: Organization ID
        project_id: Optional project ID
        retraining_job_id: Optional retraining job ID
        validation_id: Optional validation ID
        deployment_params: Optional deployment parameters
        previous_deployment_id: Optional previous deployment ID (for rollbacks)
        is_rollback: Whether this is a rollback deployment
        
    Returns:
        Created model deployment
    """
    # Create model deployment
    deployment = ModelDeployment(
        model_name=model_name,
        model_version=model_version,
        status=JobStatus.PENDING,
        deployment_params=deployment_params or {},
        retraining_job_id=retraining_job_id,
        validation_id=validation_id,
        integration_id=integration_id,
        organization_id=organization_id,
        project_id=project_id,
        previous_deployment_id=previous_deployment_id,
        is_rollback=is_rollback,
        started_at=datetime.utcnow()
    )
    
    db.add(deployment)
    db.commit()
    db.refresh(deployment)
    
    # Start deployment in background (in a real implementation, this would be a Celery task)
    # For now, we'll just update the deployment status
    deployment.status = JobStatus.RUNNING
    db.commit()
    
    # In a real implementation, we would start the deployment process here
    # For now, we'll just simulate it
    
    return deployment


def complete_model_deployment(
    db: Session,
    deployment_id: int,
    status: JobStatus,
    endpoint_url: Optional[str] = None,
    logs: Optional[str] = None
) -> ModelDeployment:
    """
    Complete a model deployment.
    
    Args:
        db: Database session
        deployment_id: ID of the model deployment
        status: Final status of the deployment
        endpoint_url: Optional endpoint URL
        logs: Optional deployment logs
        
    Returns:
        Updated model deployment
    """
    deployment = db.query(ModelDeployment).filter(ModelDeployment.id == deployment_id).first()
    
    if not deployment:
        raise ValueError(f"Model deployment with ID {deployment_id} not found")
        
    deployment.status = status
    deployment.endpoint_url = endpoint_url
    deployment.logs = logs
    deployment.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(deployment)
    
    # Trigger webhook
    if status == JobStatus.COMPLETED:
        on_model_deployed(db, deployment)
        
    return deployment


def rollback_deployment(
    db: Session,
    deployment_id: int,
    organization_id: int,
    project_id: Optional[int] = None
) -> ModelDeployment:
    """
    Rollback to a previous deployment.
    
    Args:
        db: Database session
        deployment_id: ID of the deployment to rollback to
        organization_id: Organization ID
        project_id: Optional project ID
        
    Returns:
        Created rollback deployment
    """
    # Get the deployment to rollback to
    target_deployment = db.query(ModelDeployment).filter(
        ModelDeployment.id == deployment_id,
        ModelDeployment.organization_id == organization_id
    ).first()
    
    if not target_deployment:
        raise ValueError(f"Deployment with ID {deployment_id} not found")
        
    # Get the current active deployment
    current_deployment = db.query(ModelDeployment).filter(
        ModelDeployment.model_name == target_deployment.model_name,
        ModelDeployment.organization_id == organization_id,
        ModelDeployment.status == JobStatus.COMPLETED
    ).order_by(desc(ModelDeployment.completed_at)).first()
    
    if current_deployment and current_deployment.id == deployment_id:
        # Already at this deployment
        return current_deployment
        
    # Create a rollback deployment
    rollback = deploy_model(
        db=db,
        model_name=target_deployment.model_name,
        model_version=target_deployment.model_version,
        integration_id=target_deployment.integration_id,
        organization_id=organization_id,
        project_id=project_id,
        deployment_params=target_deployment.deployment_params,
        previous_deployment_id=current_deployment.id if current_deployment else None,
        is_rollback=True
    )
    
    return rollback


def get_automl_suggestions(
    db: Session,
    model_name: str,
    organization_id: int,
    project_id: Optional[int] = None,
    days_to_analyze: int = 30
) -> List[Dict[str, Any]]:
    """
    Generate AutoML suggestions based on historical data and performance trends.
    
    Args:
        db: Database session
        model_name: Name of the model
        organization_id: Organization ID
        project_id: Optional project ID
        days_to_analyze: Number of days of data to analyze
        
    Returns:
        List of suggestions
    """
    suggestions = []
    
    # Get historical drift metrics
    drift_query = db.query(DriftMetrics).filter(
        DriftMetrics.model_name == model_name,
        DriftMetrics.organization_id == organization_id,
        DriftMetrics.timestamp >= datetime.utcnow() - timedelta(days=days_to_analyze)
    )
    
    if project_id:
        drift_query = drift_query.filter(DriftMetrics.project_id == project_id)
        
    drift_metrics = drift_query.order_by(DriftMetrics.timestamp).all()
    
    # Get historical inference logs
    logs_query = db.query(InferenceLog).filter(
        InferenceLog.model_name == model_name,
        InferenceLog.organization_id == organization_id,
        InferenceLog.timestamp >= datetime.utcnow() - timedelta(days=days_to_analyze)
    )
    
    if project_id:
        logs_query = logs_query.filter(InferenceLog.project_id == project_id)
        
    logs = logs_query.order_by(InferenceLog.timestamp).all()
    
    # Analyze drift patterns
    if drift_metrics:
        # Check for increasing drift trend
        drift_trend = [
            (m.input_kl_divergence or 0) + (m.output_kl_divergence or 0)
            for m in drift_metrics
        ]
        
        if len(drift_trend) >= 3 and all(drift_trend[i] < drift_trend[i+1] for i in range(len(drift_trend)-1)):
            # Increasing drift trend
            suggestions.append({
                "type": "hyperparameter",
                "title": "Adjust regularization to combat drift",
                "description": "Increasing data drift detected. Consider increasing regularization parameters to make the model more robust.",
                "params": {
                    "regularization_increase": 0.1
                }
            })
            
        # Check for specific feature drift
        if any(m.input_distribution_current and m.input_distribution_reference for m in drift_metrics):
            latest_drift = drift_metrics[-1]
            
            if latest_drift.input_distribution_current and latest_drift.input_distribution_reference:
                # This is a simplified example - in a real implementation, we would do more sophisticated analysis
                suggestions.append({
                    "type": "feature_engineering",
                    "title": "Feature transformation to address drift",
                    "description": "Consider applying normalization or scaling to input features to reduce sensitivity to drift.",
                    "params": {
                        "normalization": "z-score"
                    }
                })
    
    # Analyze performance patterns
    if logs:
        # Check for performance degradation
        latency_trend = [log.latency_ms for log in logs if log.latency_ms is not None]
        confidence_trend = [log.confidence for log in logs if log.confidence is not None]
        
        if latency_trend and len(latency_trend) >= 10:
            avg_latency_first_half = sum(latency_trend[:len(latency_trend)//2]) / (len(latency_trend)//2)
            avg_latency_second_half = sum(latency_trend[len(latency_trend)//2:]) / (len(latency_trend) - len(latency_trend)//2)
            
            if avg_latency_second_half > avg_latency_first_half * 1.2:  # 20% increase
                suggestions.append({
                    "type": "optimization",
                    "title": "Model optimization for latency",
                    "description": f"Latency has increased by {((avg_latency_second_half / avg_latency_first_half) - 1) * 100:.1f}%. Consider model quantization or pruning.",
                    "params": {
                        "quantization": "int8",
                        "pruning_factor": 0.3
                    }
                })
                
        if confidence_trend and len(confidence_trend) >= 10:
            avg_confidence_first_half = sum(confidence_trend[:len(confidence_trend)//2]) / (len(confidence_trend)//2)
            avg_confidence_second_half = sum(confidence_trend[len(confidence_trend)//2:]) / (len(confidence_trend) - len(confidence_trend)//2)
            
            if avg_confidence_second_half < avg_confidence_first_half * 0.9:  # 10% decrease
                suggestions.append({
                    "type": "architecture",
                    "title": "Model architecture upgrade",
                    "description": f"Prediction confidence has decreased by {(1 - (avg_confidence_second_half / avg_confidence_first_half)) * 100:.1f}%. Consider upgrading to a more powerful model architecture.",
                    "params": {
                        "architecture": "transformer",
                        "layers": 12,
                        "attention_heads": 8
                    }
                })
    
    # Add general suggestions based on model type and data characteristics
    # This would be more sophisticated in a real implementation
    suggestions.append({
        "type": "training",
        "title": "Experiment with learning rate schedules",
        "description": "Try using a cyclical learning rate schedule to improve convergence and generalization.",
        "params": {
            "lr_schedule": "cyclical",
            "base_lr": 0.001,
            "max_lr": 0.01,
            "step_size": 1000
        }
    })
    
    return suggestions