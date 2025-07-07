"""
Webhooks module for triggering external workflows from ModelPulse events.

This module provides functionality to:
- Manage webhooks (create, update, delete, get)
- Trigger webhooks on specific events
- Track webhook invocation history
"""

from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session
from fastapi import HTTPException
import json
import logging
import requests
from datetime import datetime
import asyncio
import aiohttp

from .models import Webhook, WebhookHistory, WebhookEvent
from .schemas import WebhookCreate, WebhookUpdate

# Set up logging
logger = logging.getLogger(__name__)


async def _send_webhook_async(webhook: Webhook, event: WebhookEvent, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a webhook asynchronously and return the result."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ModelPulse-Webhook",
        "X-ModelPulse-Event": event
    }

    # Add custom headers if provided
    if webhook.headers:
        headers.update(webhook.headers)

    # Add event info to payload
    full_payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "data": payload
    }

    result = {
        "webhook_id": webhook.id,
        "event": event,
        "payload": full_payload,
        "created_at": datetime.utcnow()
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                str(webhook.url), 
                headers=headers, 
                json=full_payload,
                timeout=10
            ) as response:
                result["response_status"] = response.status
                result["response_body"] = await response.text()

                if response.status < 200 or response.status >= 300:
                    result["error"] = f"Webhook returned non-success status code: {response.status}"
                    logger.warning(f"Webhook {webhook.id} failed with status {response.status}: {result['response_body']}")
                else:
                    logger.info(f"Webhook {webhook.id} triggered successfully for event {event}")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error triggering webhook {webhook.id}: {str(e)}")

    return result


def _record_webhook_history(db: Session, result: Dict[str, Any]) -> WebhookHistory:
    """Record webhook invocation history."""
    history = WebhookHistory(
        webhook_id=result["webhook_id"],
        event=result["event"],
        payload=result["payload"],
        response_status=result.get("response_status"),
        response_body=result.get("response_body"),
        error=result.get("error")
    )

    db.add(history)
    db.commit()
    db.refresh(history)
    return history


def create_webhook(db: Session, webhook: WebhookCreate) -> Webhook:
    """Create a new webhook."""
    db_webhook = Webhook(
        name=webhook.name,
        url=str(webhook.url),
        events=webhook.events,
        headers=webhook.headers,
        is_active=webhook.is_active,
        organization_id=webhook.organization_id,
        project_id=webhook.project_id
    )

    # Test the URL by sending a test event
    try:
        response = requests.post(
            str(webhook.url),
            headers={"Content-Type": "application/json", "User-Agent": "ModelPulse-Webhook-Test"},
            json={"event": "test", "timestamp": datetime.utcnow().isoformat(), "data": {"message": "Test webhook"}},
            timeout=5
        )

        if response.status_code < 200 or response.status_code >= 300:
            logger.warning(f"Webhook test returned non-success status code: {response.status_code}")
            # We still create the webhook, but log a warning
    except Exception as e:
        logger.warning(f"Webhook test failed: {str(e)}")
        # We still create the webhook, but log a warning

    db.add(db_webhook)
    db.commit()
    db.refresh(db_webhook)
    return db_webhook


def update_webhook(db: Session, webhook_id: int, webhook_update: WebhookUpdate) -> Webhook:
    """Update an existing webhook."""
    db_webhook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not db_webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    update_data = webhook_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_webhook, key, value)

    # If URL was updated, test it
    if "url" in update_data:
        try:
            response = requests.post(
                str(db_webhook.url),
                headers={"Content-Type": "application/json", "User-Agent": "ModelPulse-Webhook-Test"},
                json={"event": "test", "timestamp": datetime.utcnow().isoformat(), "data": {"message": "Test webhook"}},
                timeout=5
            )

            if response.status_code < 200 or response.status_code >= 300:
                logger.warning(f"Webhook test returned non-success status code: {response.status_code}")
                # We still update the webhook, but log a warning
        except Exception as e:
            logger.warning(f"Webhook test failed: {str(e)}")
            # We still update the webhook, but log a warning

    db.commit()
    db.refresh(db_webhook)
    return db_webhook


def delete_webhook(db: Session, webhook_id: int) -> bool:
    """Delete a webhook."""
    db_webhook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not db_webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    db.delete(db_webhook)
    db.commit()
    return True


def get_webhook(db: Session, webhook_id: int) -> Webhook:
    """Get a webhook by ID."""
    db_webhook = db.query(Webhook).filter(Webhook.id == webhook_id).first()
    if not db_webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return db_webhook


def get_webhooks(db: Session, organization_id: int, project_id: Optional[int] = None, 
                event: Optional[WebhookEvent] = None) -> List[Webhook]:
    """Get webhooks by organization ID and optionally project ID or event."""
    query = db.query(Webhook).filter(Webhook.organization_id == organization_id)

    if project_id:
        query = query.filter(Webhook.project_id == project_id)

    webhooks = query.all()

    # Filter by event if specified
    if event:
        webhooks = [w for w in webhooks if event in w.events]

    return webhooks


def get_webhook_history(db: Session, webhook_id: int, limit: int = 100) -> List[WebhookHistory]:
    """Get webhook invocation history."""
    return db.query(WebhookHistory).filter(
        WebhookHistory.webhook_id == webhook_id
    ).order_by(WebhookHistory.created_at.desc()).limit(limit).all()


async def trigger_webhooks(db: Session, event: WebhookEvent, payload: Dict[str, Any], 
                         organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Trigger all webhooks for a specific event."""
    # Get all active webhooks for this organization that are subscribed to this event
    webhooks = get_webhooks(db, organization_id, project_id, event)
    webhooks = [w for w in webhooks if w.is_active]

    if not webhooks:
        return []

    # Send webhooks asynchronously
    tasks = [_send_webhook_async(webhook, event, payload) for webhook in webhooks]
    results = await asyncio.gather(*tasks)

    # Record history for each webhook
    history_entries = [_record_webhook_history(db, result) for result in results]

    return history_entries


def trigger_webhooks_sync(db: Session, event: WebhookEvent, payload: Dict[str, Any],
                        organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Synchronous version of trigger_webhooks for use in synchronous contexts."""
    # Create an event loop if there isn't one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        trigger_webhooks(db, event, payload, organization_id, project_id)
    )


# Event handler functions that can be called from other parts of the application

def on_drift_detected(db: Session, model_name: str, drift_metrics: Dict[str, Any], 
                     organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Handle drift detected event."""
    payload = {
        "model_name": model_name,
        "drift_metrics": drift_metrics
    }
    return trigger_webhooks_sync(db, WebhookEvent.DRIFT_DETECTED, payload, organization_id, project_id)


def on_threshold_breached(db: Session, model_name: str, metric_name: str, threshold_value: float,
                         actual_value: float, organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Handle threshold breached event."""
    payload = {
        "model_name": model_name,
        "metric_name": metric_name,
        "threshold_value": threshold_value,
        "actual_value": actual_value
    }
    return trigger_webhooks_sync(db, WebhookEvent.THRESHOLD_BREACHED, payload, organization_id, project_id)


def on_recommendation_created(db: Session, recommendation: Dict[str, Any],
                            organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Handle recommendation created event."""
    return trigger_webhooks_sync(db, WebhookEvent.RECOMMENDATION_CREATED, recommendation, organization_id, project_id)


def on_model_registered(db: Session, model_name: str, model_version: str, metadata: Dict[str, Any],
                       organization_id: int, project_id: Optional[int] = None) -> List[WebhookHistory]:
    """Handle model registered event."""
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "metadata": metadata
    }
    return trigger_webhooks_sync(db, WebhookEvent.MODEL_REGISTERED, payload, organization_id, project_id)


def on_retraining_started(db: Session, job: Any) -> List[WebhookHistory]:
    """Handle retraining started event."""
    payload = {
        "job_id": job.id,
        "model_name": job.model_name,
        "organization_id": job.organization_id,
        "project_id": job.project_id,
        "policy_id": job.policy_id,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "training_params": job.training_params
    }
    return trigger_webhooks_sync(db, WebhookEvent.RETRAINING_STARTED, payload, job.organization_id, job.project_id)


def on_retraining_completed(db: Session, job: Any) -> List[WebhookHistory]:
    """Handle retraining completed event."""
    payload = {
        "job_id": job.id,
        "model_name": job.model_name,
        "organization_id": job.organization_id,
        "project_id": job.project_id,
        "policy_id": job.policy_id,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "metrics": job.metrics
    }
    return trigger_webhooks_sync(db, WebhookEvent.RETRAINING_COMPLETED, payload, job.organization_id, job.project_id)


def on_validation_failed(db: Session, validation: Any) -> List[WebhookHistory]:
    """Handle validation failed event."""
    payload = {
        "validation_id": validation.id,
        "model_name": validation.model_name,
        "model_version": validation.model_version,
        "organization_id": validation.organization_id,
        "project_id": validation.project_id,
        "retraining_job_id": validation.retraining_job_id,
        "status": validation.status,
        "started_at": validation.started_at.isoformat() if validation.started_at else None,
        "completed_at": validation.completed_at.isoformat() if validation.completed_at else None,
        "metrics": validation.metrics
    }
    return trigger_webhooks_sync(db, WebhookEvent.VALIDATION_FAILED, payload, validation.organization_id, validation.project_id)


def on_model_deployed(db: Session, deployment: Any) -> List[WebhookHistory]:
    """Handle model deployed event."""
    payload = {
        "deployment_id": deployment.id,
        "model_name": deployment.model_name,
        "model_version": deployment.model_version,
        "organization_id": deployment.organization_id,
        "project_id": deployment.project_id,
        "retraining_job_id": deployment.retraining_job_id,
        "validation_id": deployment.validation_id,
        "integration_id": deployment.integration_id,
        "status": deployment.status,
        "endpoint_url": deployment.endpoint_url,
        "is_rollback": deployment.is_rollback,
        "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
        "completed_at": deployment.completed_at.isoformat() if deployment.completed_at else None
    }
    return trigger_webhooks_sync(db, WebhookEvent.MODEL_DEPLOYED, payload, deployment.organization_id, deployment.project_id)
