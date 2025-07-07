from datetime import datetime
from typing import Optional, Dict, Any, Callable
from fastapi import Request, Response
from sqlalchemy.orm import Session
import json
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .models import AuditLog, AuditActionType
from .database import get_db
from .schemas import AuditLogCreate

class AuditLogger:
    """
    Utility class for logging audit events
    """
    @staticmethod
    async def log_action(
        db: Session,
        action: AuditActionType,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[int] = None,
        organization_id: Optional[int] = None,
        project_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None
    ) -> AuditLog:
        """
        Log an action to the audit log
        """
        audit_log = AuditLog(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            organization_id=organization_id,
            project_id=project_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            status=status,
            timestamp=datetime.utcnow()
        )
        
        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)
        return audit_log

class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging audit events for specific endpoints
    """
    def __init__(
        self, 
        app: ASGIApp,
        audit_endpoints: Dict[str, Dict[str, Any]] = None
    ):
        super().__init__(app)
        # Dictionary mapping endpoints to audit configuration
        # Example: {"/users": {"GET": {"action": "access", "resource_type": "user"}}}
        self.audit_endpoints = audit_endpoints or {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get path and method
        path = request.url.path
        method = request.method
        
        # Check if this endpoint should be audited
        should_audit = False
        audit_config = None
        
        for endpoint, methods in self.audit_endpoints.items():
            if path.startswith(endpoint) and method in methods:
                should_audit = True
                audit_config = methods[method]
                break
        
        # Process the request
        response = await call_next(request)
        
        # If this endpoint should be audited, log the action
        if should_audit:
            try:
                # Get user ID from request state if available
                user_id = getattr(request.state, "user_id", None)
                organization_id = getattr(request.state, "organization_id", None)
                
                # Get client info
                ip_address = request.client.host if request.client else None
                user_agent = request.headers.get("user-agent")
                
                # Get response status
                status = "success" if response.status_code < 400 else "failure"
                
                # Create details with request and response info
                details = {
                    "request_path": str(request.url),
                    "request_method": method,
                    "response_status": response.status_code,
                }
                
                # Get DB session
                db = next(get_db())
                
                # Log the action
                await AuditLogger.log_action(
                    db=db,
                    action=audit_config.get("action", AuditActionType.ACCESS),
                    resource_type=audit_config.get("resource_type", "unknown"),
                    resource_id=audit_config.get("resource_id_extractor", lambda r: None)(request),
                    user_id=user_id,
                    organization_id=organization_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details=details,
                    status=status
                )
            except Exception as e:
                # Log error but don't fail the request
                print(f"Error logging audit event: {str(e)}")
        
        return response

# Function to create audit middleware with configuration
def create_audit_middleware():
    """
    Create an instance of AuditMiddleware with predefined audit endpoints
    """
    # Define which endpoints to audit and how
    audit_endpoints = {
        "/token": {
            "POST": {
                "action": AuditActionType.LOGIN,
                "resource_type": "user",
                "resource_id_extractor": lambda r: None  # Extract from form data if needed
            }
        },
        "/users": {
            "POST": {
                "action": AuditActionType.CREATE,
                "resource_type": "user",
                "resource_id_extractor": lambda r: None
            },
            "GET": {
                "action": AuditActionType.ACCESS,
                "resource_type": "user",
                "resource_id_extractor": lambda r: r.path_params.get("user_id")
            }
        },
        "/organizations": {
            "POST": {
                "action": AuditActionType.CREATE,
                "resource_type": "organization",
                "resource_id_extractor": lambda r: None
            },
            "PUT": {
                "action": AuditActionType.UPDATE,
                "resource_type": "organization",
                "resource_id_extractor": lambda r: r.path_params.get("organization_id")
            },
            "DELETE": {
                "action": AuditActionType.DELETE,
                "resource_type": "organization",
                "resource_id_extractor": lambda r: r.path_params.get("organization_id")
            }
        },
        # Add more endpoints as needed
    }
    
    return lambda app: AuditMiddleware(app, audit_endpoints)

# Helper functions for manual audit logging
async def log_user_login(db: Session, user_id: int, ip_address: Optional[str] = None, user_agent: Optional[str] = None, status: str = "success"):
    """Log a user login event"""
    return await AuditLogger.log_action(
        db=db,
        action=AuditActionType.LOGIN,
        resource_type="user",
        resource_id=str(user_id),
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        status=status
    )

async def log_user_logout(db: Session, user_id: int, ip_address: Optional[str] = None, user_agent: Optional[str] = None):
    """Log a user logout event"""
    return await AuditLogger.log_action(
        db=db,
        action=AuditActionType.LOGOUT,
        resource_type="user",
        resource_id=str(user_id),
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        status="success"
    )

async def log_config_change(db: Session, user_id: int, resource_type: str, resource_id: str, details: Dict[str, Any], organization_id: Optional[int] = None):
    """Log a configuration change event"""
    return await AuditLogger.log_action(
        db=db,
        action=AuditActionType.CONFIG_CHANGE,
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user_id,
        organization_id=organization_id,
        details=details,
        status="success"
    )

async def log_data_export(db: Session, user_id: int, resource_type: str, details: Dict[str, Any], organization_id: Optional[int] = None):
    """Log a data export event"""
    return await AuditLogger.log_action(
        db=db,
        action=AuditActionType.EXPORT,
        resource_type=resource_type,
        user_id=user_id,
        organization_id=organization_id,
        details=details,
        status="success"
    )