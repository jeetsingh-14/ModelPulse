from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta, date
import time
from collections import defaultdict
import csv
import os
import io
from tempfile import NamedTemporaryFile
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from prometheus_fastapi_instrumentator import Instrumentator

from .models import (
    InferenceLog,
    AlertThreshold,
    DriftMetrics,
    DriftSeverity,
    User,
    UserRole,
    Organization,
    Project,
    OrganizationRole,
    Recommendation,
    RecommendationAction,
    RecommendationPriority,
)
from .drift import compute_drift_metrics, get_drift_summary
from .predictive import (
    predict_drift_metrics,
    predict_performance_metrics,
    get_all_predictive_alerts,
)
from .recommendations import (
    generate_drift_recommendations,
    generate_performance_recommendations,
    get_recommendations,
    generate_all_recommendations,
)
from .assistant import chat_with_assistant
from .auth import (
    create_access_token,
    get_current_user,
    get_current_active_user,
    get_admin_user,
    get_analyst_user,
    get_organization_member,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from .schemas import (
    InferenceLogCreate,
    InferenceLogResponse,
    AlertThresholdCreate,
    AlertThresholdUpdate,
    AlertThresholdResponse,
    Alert,
    AnalyticsSummary,
    DriftMetricsCreate,
    DriftMetricsResponse,
    DriftSummary,
    UserCreate,
    UserUpdate,
    UserResponse,
    Token,
    TokenData,
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse,
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    OrganizationRoleCreate,
    OrganizationRoleUpdate,
    OrganizationRoleResponse,
    OrganizationInvitationCreate,
    OrganizationInvitationResponse,
    ForecastPoint,
    DriftPrediction,
    PerformancePrediction,
    PredictiveAlert,
    Recommendation,
    RecommendationCreate,
    RecommendationUpdate,
    RecommendationResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
)
from .database import get_db, engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize Sentry for error monitoring
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),  # Set your Sentry DSN in environment variables
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,  # Capture 100% of transactions for performance monitoring
    environment=os.getenv("ENVIRONMENT", "development"),
)

app = FastAPI(
    title="ModelPulse API", description="Real-time ML model monitoring platform"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Prometheus metrics
instrumentator = Instrumentator().instrument(app)

@app.on_event("startup")
async def startup():
    instrumentator.expose(app, include_in_schema=False, should_gzip=True)


@app.get("/")
def read_root():
    return {"message": "Hello from ModelPulse"}


# Authentication Endpoints
@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db),
    organization_id: Optional[int] = Query(None, description="Organization ID for tenant context")
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # Authenticate user
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not user.verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Default to global role
    org_role = None

    # If organization_id is provided, check if user is a member and get their role
    if organization_id:
        # Check if user is a member of the organization
        org_role_record = db.query(OrganizationRole).filter(
            OrganizationRole.user_id == user.id,
            OrganizationRole.organization_id == organization_id
        ).first()

        if not org_role_record:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not a member of the specified organization",
            )

        org_role = org_role_record.role

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires,
        organization_id=organization_id,
        org_role=org_role
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users", response_model=UserResponse)
def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    """
    Create a new user (admin only).
    """
    # Check if username or email already exists
    db_user = (
        db.query(User)
        .filter((User.username == user.username) | (User.email == user.email))
        .first()
    )

    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    # Create new user
    hashed_password = User.get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        hashed_password=hashed_password,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@app.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user (public endpoint, creates viewer users only).
    """
    # Check if username or email already exists
    db_user = (
        db.query(User)
        .filter((User.username == user.username) | (User.email == user.email))
        .first()
    )

    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    # Create new user with viewer role only
    hashed_password = User.get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=UserRole.VIEWER,  # Force viewer role for public registration
        hashed_password=hashed_password,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user


@app.get("/users", response_model=List[UserResponse])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    """
    Get all users (admin only).
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update a user. Admins can update any user, users can only update themselves.
    """
    db_user = db.query(User).filter(User.id == user_id).first()

    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Check permissions: only admins can update other users
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update other users",
        )

    # Only admins can change roles
    if user_update.role is not None and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to change user role",
        )

    # Update user fields
    update_data = user_update.dict(exclude_unset=True)

    # Hash password if it's being updated
    if "password" in update_data:
        update_data["hashed_password"] = User.get_password_hash(
            update_data.pop("password")
        )

    for key, value in update_data.items():
        setattr(db_user, key, value)

    db.commit()
    db.refresh(db_user)

    return db_user


@app.delete("/users/{user_id}", response_model=Dict[str, Any])
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),
):
    """
    Delete a user (admin only).
    """
    db_user = db.query(User).filter(User.id == user_id).first()

    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent deleting yourself
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own user account",
        )

    db.delete(db_user)
    db.commit()

    return {"message": "User deleted successfully"}


@app.post("/log", response_model=Dict[str, Any])
def log_inference(
    log: InferenceLogCreate,
    project_id: Optional[int] = Query(None, description="Project ID for the log"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),  # Require organization membership
):
    # Check if the user has a current organization context
    if not hasattr(current_user, 'current_organization_id') or not current_user.current_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No organization context provided",
        )

    # Check if the user has analyst or admin role in the organization
    if current_user.current_org_role not in [UserRole.ADMIN, UserRole.ANALYST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and analysts can log inferences",
        )

    # If project_id is provided, verify it belongs to the current organization
    if project_id:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.organization_id == current_user.current_organization_id
        ).first()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or does not belong to the current organization",
            )

    # Create the inference log with organization context
    db_log = InferenceLog(
        model_name=log.model_name,
        timestamp=log.timestamp,
        input_shape=log.input_shape,
        latency_ms=log.latency_ms,
        confidence=log.confidence,
        output_class=log.output_class,
        organization_id=current_user.current_organization_id,
        project_id=project_id,
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    # Check for threshold breaches
    alerts = []

    # Get active thresholds that apply to this model or all models, scoped to the organization
    thresholds_query = (
        db.query(AlertThreshold)
        .filter(
            (AlertThreshold.model_name == log.model_name)
            | (AlertThreshold.model_name == None)
        )
        .filter(AlertThreshold.is_active == True)
        .filter(AlertThreshold.organization_id == current_user.current_organization_id)
    )

    # If project_id is provided, include thresholds for that project or global thresholds
    if project_id:
        thresholds_query = thresholds_query.filter(
            (AlertThreshold.project_id == project_id) | (AlertThreshold.project_id == None)
        )
    else:
        thresholds_query = thresholds_query.filter(AlertThreshold.project_id == None)

    thresholds = thresholds_query.all()

    for threshold in thresholds:
        # Check if the metric exists in the log
        if (
            threshold.metric_name == "latency_ms"
            and log.latency_ms > threshold.threshold_value
        ):
            alerts.append(
                Alert(
                    model_name=log.model_name,
                    metric_name="latency_ms",
                    threshold_value=threshold.threshold_value,
                    actual_value=log.latency_ms,
                    timestamp=log.timestamp,
                )
            )
        elif (
            threshold.metric_name == "confidence"
            and log.confidence < threshold.threshold_value
        ):
            alerts.append(
                Alert(
                    model_name=log.model_name,
                    metric_name="confidence",
                    threshold_value=threshold.threshold_value,
                    actual_value=log.confidence,
                    timestamp=log.timestamp,
                )
            )

    return {"log": db_log, "alerts": alerts}


@app.get("/logs", response_model=List[InferenceLogResponse])
def get_logs(
    model_name: Optional[str] = None,
    output_class: Optional[str] = None,
    project_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),  # Require organization membership
):
    """
    Get inference logs with optional filtering by model, output class, project, and date range.
    Logs are scoped to the current organization.
    """
    # Check if the user has a current organization context
    if not hasattr(current_user, 'current_organization_id') or not current_user.current_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No organization context provided",
        )

    # Start query with organization filter
    query = db.query(InferenceLog).filter(
        InferenceLog.organization_id == current_user.current_organization_id
    )

    # Apply optional filters
    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)

    if output_class:
        query = query.filter(InferenceLog.output_class == output_class)

    if project_id:
        # Verify project belongs to the organization
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.organization_id == current_user.current_organization_id
        ).first()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or does not belong to the current organization",
            )

        query = query.filter(InferenceLog.project_id == project_id)

    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(InferenceLog.timestamp >= start_datetime)

    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(InferenceLog.timestamp <= end_datetime)

    # Order by timestamp descending (newest first)
    query = query.order_by(InferenceLog.timestamp.desc())

    logs = query.offset(skip).limit(limit).all()
    return logs


@app.get("/logs/export")
def export_logs(
    model_name: Optional[str] = None,
    output_class: Optional[str] = None,
    project_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    format: str = "csv",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),  # Require organization membership
):
    """
    Export inference logs to CSV format with optional filtering.
    Logs are scoped to the current organization.
    """
    # Check if the user has a current organization context
    if not hasattr(current_user, 'current_organization_id') or not current_user.current_organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No organization context provided",
        )

    # Start query with organization filter
    query = db.query(InferenceLog).filter(
        InferenceLog.organization_id == current_user.current_organization_id
    )

    # Apply optional filters
    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)

    if output_class:
        query = query.filter(InferenceLog.output_class == output_class)

    if project_id:
        # Verify project belongs to the organization
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.organization_id == current_user.current_organization_id
        ).first()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or does not belong to the current organization",
            )

        query = query.filter(InferenceLog.project_id == project_id)

    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(InferenceLog.timestamp >= start_datetime)

    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(InferenceLog.timestamp <= end_datetime)

    # Order by timestamp
    query = query.order_by(InferenceLog.timestamp.desc())

    # Get all logs matching the filters
    logs = query.all()

    if not logs:
        raise HTTPException(
            status_code=404, detail="No logs found matching the criteria"
        )

    # Create a temporary file
    with NamedTemporaryFile(
        delete=False, suffix=f".{format}", mode="w+", newline=""
    ) as temp_file:
        # Create CSV writer
        fieldnames = [
            "id",
            "model_name",
            "timestamp",
            "input_shape",
            "latency_ms",
            "confidence",
            "output_class",
        ]
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write logs to CSV
        for log in logs:
            writer.writerow(
                {
                    "id": log.id,
                    "model_name": log.model_name,
                    "timestamp": log.timestamp.isoformat(),
                    "input_shape": str(log.input_shape),
                    "latency_ms": log.latency_ms,
                    "confidence": log.confidence,
                    "output_class": log.output_class,
                }
            )

        temp_file_path = temp_file.name

    # Generate a meaningful filename
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modelpulse_logs_{date_str}.{format}"

    # Return the file
    return FileResponse(
        path=temp_file_path,
        filename=filename,
        media_type="text/csv",
        background=BackgroundTasks().add_task(lambda: os.unlink(temp_file_path)),
    )


# Alert Threshold Endpoints
@app.post("/alert-thresholds", response_model=AlertThresholdResponse)
def create_alert_threshold(
    threshold: AlertThresholdCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),  # Admin only
):
    db_threshold = AlertThreshold(
        model_name=threshold.model_name,
        metric_name=threshold.metric_name,
        threshold_value=threshold.threshold_value,
        is_active=threshold.is_active,
    )
    db.add(db_threshold)
    db.commit()
    db.refresh(db_threshold)
    return db_threshold


@app.get("/alert-thresholds", response_model=List[AlertThresholdResponse])
def get_alert_thresholds(
    model_name: Optional[str] = None,
    metric_name: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    query = db.query(AlertThreshold)

    if model_name:
        query = query.filter(AlertThreshold.model_name == model_name)

    if metric_name:
        query = query.filter(AlertThreshold.metric_name == metric_name)

    if is_active is not None:
        query = query.filter(AlertThreshold.is_active == is_active)

    thresholds = query.all()
    return thresholds


@app.get("/alert-thresholds/{threshold_id}", response_model=AlertThresholdResponse)
def get_alert_threshold(
    threshold_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    threshold = (
        db.query(AlertThreshold).filter(AlertThreshold.id == threshold_id).first()
    )
    if threshold is None:
        raise HTTPException(status_code=404, detail="Alert threshold not found")
    return threshold


@app.put("/alert-thresholds/{threshold_id}", response_model=AlertThresholdResponse)
def update_alert_threshold(
    threshold_id: int,
    threshold_update: AlertThresholdUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),  # Admin only
):
    db_threshold = (
        db.query(AlertThreshold).filter(AlertThreshold.id == threshold_id).first()
    )
    if db_threshold is None:
        raise HTTPException(status_code=404, detail="Alert threshold not found")

    update_data = threshold_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_threshold, key, value)

    db_threshold.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_threshold)
    return db_threshold


@app.delete("/alert-thresholds/{threshold_id}", response_model=Dict[str, Any])
def delete_alert_threshold(
    threshold_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_admin_user),  # Admin only
):
    db_threshold = (
        db.query(AlertThreshold).filter(AlertThreshold.id == threshold_id).first()
    )
    if db_threshold is None:
        raise HTTPException(status_code=404, detail="Alert threshold not found")

    db.delete(db_threshold)
    db.commit()
    return {"message": "Alert threshold deleted successfully"}


@app.get("/analytics", response_model=AnalyticsSummary)
def get_analytics(
    model_name: Optional[str] = None,
    time_range: Optional[int] = 24,  # Hours
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    # Set time filter based on either time_range or start_date/end_date
    if start_date and end_date:
        # Convert dates to datetime with time at start/end of day
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = db.query(InferenceLog).filter(
            InferenceLog.timestamp >= start_datetime,
            InferenceLog.timestamp <= end_datetime,
        )
    else:
        # Use time_range as before
        time_filter = datetime.utcnow() - timedelta(hours=time_range)
        query = db.query(InferenceLog).filter(InferenceLog.timestamp >= time_filter)

    # Apply model filter if provided
    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)

    # Get all logs for the period
    logs = query.all()

    # If no logs, return empty analytics
    if not logs:
        return AnalyticsSummary(
            avg_latency=0,
            avg_confidence=0,
            total_requests=0,
            requests_per_minute=0,
            model_distribution={},
            class_distribution={},
            latency_over_time=[],
            confidence_distribution=[],
        )

    # Calculate average latency and confidence
    avg_latency = sum(log.latency_ms for log in logs) / len(logs)
    avg_confidence = sum(log.confidence for log in logs) / len(logs)

    # Calculate total requests
    total_requests = len(logs)

    # Calculate requests per minute
    if start_date and end_date:
        time_span_minutes = (end_datetime - start_datetime).total_seconds() / 60
    else:
        time_span_minutes = time_range * 60  # Convert hours to minutes

    requests_per_minute = (
        total_requests / time_span_minutes if time_span_minutes > 0 else 0
    )

    # Calculate model distribution
    model_distribution = defaultdict(int)
    for log in logs:
        model_distribution[log.model_name] += 1

    # Calculate class distribution
    class_distribution = defaultdict(int)
    for log in logs:
        class_distribution[log.output_class] += 1

    # Prepare latency over time data
    # Group by hour and calculate average latency
    latency_over_time = []
    hour_groups = defaultdict(list)

    for log in logs:
        hour_key = log.timestamp.replace(minute=0, second=0, microsecond=0)
        hour_groups[hour_key].append(log.latency_ms)

    for hour, latencies in sorted(hour_groups.items()):
        latency_over_time.append(
            {
                "timestamp": hour.isoformat(),
                "avg_latency": sum(latencies) / len(latencies),
            }
        )

    # Prepare confidence distribution
    # Create 10 buckets from 0 to 1
    confidence_buckets = defaultdict(int)
    for log in logs:
        bucket = int(log.confidence * 10) / 10  # Round to nearest 0.1
        confidence_buckets[bucket] += 1

    confidence_distribution = [
        {"confidence_range": f"{bucket:.1f}-{bucket+0.1:.1f}", "count": count}
        for bucket, count in sorted(confidence_buckets.items())
    ]

    return AnalyticsSummary(
        avg_latency=avg_latency,
        avg_confidence=avg_confidence,
        total_requests=total_requests,
        requests_per_minute=requests_per_minute,
        model_distribution=dict(model_distribution),
        class_distribution=dict(class_distribution),
        latency_over_time=latency_over_time,
        confidence_distribution=confidence_distribution,
    )


# Drift Detection Endpoints
@app.post("/drift/compute", response_model=DriftMetricsResponse)
def compute_model_drift(
    model_name: str,
    reference_period_days: int = 7,
    current_period_days: int = 1,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user),  # Require analyst or admin role
):
    """
    Compute drift metrics for a model by comparing current data to a reference period.
    """
    # Compute drift metrics
    drift_metrics_data = compute_drift_metrics(
        db=db,
        model_name=model_name,
        reference_period_days=reference_period_days,
        current_period_days=current_period_days,
    )

    if drift_metrics_data is None:
        raise HTTPException(
            status_code=400,
            detail="Not enough data to compute drift metrics. Need at least 10 samples in both reference and current periods.",
        )

    # Create drift metrics record
    db_drift_metrics = DriftMetrics(
        model_name=drift_metrics_data.model_name,
        timestamp=drift_metrics_data.timestamp or datetime.utcnow(),
        input_kl_divergence=drift_metrics_data.input_kl_divergence,
        input_psi=drift_metrics_data.input_psi,
        input_distribution_reference=drift_metrics_data.input_distribution_reference,
        input_distribution_current=drift_metrics_data.input_distribution_current,
        output_kl_divergence=drift_metrics_data.output_kl_divergence,
        output_psi=drift_metrics_data.output_psi,
        output_distribution_reference=drift_metrics_data.output_distribution_reference,
        output_distribution_current=drift_metrics_data.output_distribution_current,
        drift_severity=drift_metrics_data.drift_severity,
        explanation=drift_metrics_data.explanation,
    )

    db.add(db_drift_metrics)
    db.commit()
    db.refresh(db_drift_metrics)

    return db_drift_metrics


@app.get("/drift", response_model=List[DriftMetricsResponse])
def get_drift_metrics(
    model_name: Optional[str] = None,
    severity: Optional[DriftSeverity] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    """
    Get drift metrics with optional filtering by model, severity, and date range.
    """
    query = db.query(DriftMetrics)

    if model_name:
        query = query.filter(DriftMetrics.model_name == model_name)

    if severity:
        query = query.filter(DriftMetrics.drift_severity == severity)

    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DriftMetrics.timestamp >= start_datetime)

    if end_date:
        end_datetime = datetime.combine(end_date, datetime.max.time())
        query = query.filter(DriftMetrics.timestamp <= end_datetime)

    # Order by timestamp descending (newest first)
    query = query.order_by(DriftMetrics.timestamp.desc())

    # Apply pagination
    drift_metrics = query.offset(skip).limit(limit).all()

    return drift_metrics


@app.get("/drift/{model_name}/summary", response_model=DriftSummary)
def get_model_drift_summary(
    model_name: str,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    """
    Get a summary of drift metrics for a specific model over time.
    """
    return get_drift_summary(db=db, model_name=model_name, days=days)


@app.get("/predictive/drift/{model_name}", response_model=DriftPrediction, tags=["Predictive Analytics"])
def get_predictive_drift(
    model_name: str, 
    forecast_horizon: int = Query(24, description="Number of hours to forecast"),
    threshold_days: int = Query(30, description="Number of days of historical data to use"),
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_analyst_user)
):
    """
    Get predictive drift metrics for a specific model.
    """
    prediction = predict_drift_metrics(db, model_name, forecast_horizon, threshold_days)
    return prediction


@app.get("/predictive/performance/{model_name}/{metric_type}", response_model=PerformancePrediction, tags=["Predictive Analytics"])
def get_predictive_performance(
    model_name: str,
    metric_type: str = Path(..., description="Type of metric to forecast (latency, confidence)"),
    forecast_horizon: int = Query(24, description="Number of hours to forecast"),
    threshold_days: int = Query(30, description="Number of days of historical data to use"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user)
):
    """
    Get predictive performance metrics for a specific model.
    """
    if metric_type not in ["latency", "confidence"]:
        raise HTTPException(status_code=400, detail=f"Invalid metric type: {metric_type}. Must be 'latency' or 'confidence'.")

    prediction = predict_performance_metrics(db, model_name, metric_type, forecast_horizon, threshold_days)
    return prediction


@app.get("/predictive/alerts", response_model=List[PredictiveAlert], tags=["Predictive Analytics"])
def get_predictive_alerts(
    model_name: Optional[str] = Query(None, description="Optional filter for specific model"),
    forecast_horizon: int = Query(24, description="Number of hours to forecast"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user)
):
    """
    Get all predictive alerts for models.
    """
    alerts = get_all_predictive_alerts(db, model_name, forecast_horizon)
    return alerts


# Recommendation Endpoints
@app.get("/recommendations", response_model=List[RecommendationResponse], tags=["Recommendations"])
def read_recommendations(
    model_name: Optional[str] = Query(None, description="Optional filter for specific model"),
    include_resolved: bool = Query(False, description="Whether to include resolved recommendations"),
    limit: int = Query(100, description="Maximum number of recommendations to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get recommendations from the database.
    """
    # Get organization ID from the current user's context
    organization_id = None
    if hasattr(current_user, "organization_id"):
        organization_id = current_user.organization_id

    recommendations = get_recommendations(
        db=db, 
        model_name=model_name, 
        organization_id=organization_id,
        include_resolved=include_resolved,
        limit=limit
    )
    return recommendations


@app.post("/recommendations/generate", response_model=List[RecommendationResponse], tags=["Recommendations"])
def generate_recommendations(
    organization_id: int = Query(..., description="Organization ID"),
    project_id: Optional[int] = Query(None, description="Optional project ID"),
    days_to_analyze: int = Query(7, description="Number of days of data to analyze"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user)
):
    """
    Generate recommendations for all models in an organization/project.
    """
    recommendations = generate_all_recommendations(
        db=db,
        organization_id=organization_id,
        project_id=project_id,
        days_to_analyze=days_to_analyze
    )
    return recommendations


@app.put("/recommendations/{recommendation_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def update_recommendation(
    recommendation_id: int,
    recommendation_update: RecommendationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user)
):
    """
    Update a recommendation (e.g., mark as resolved).
    """
    recommendation = db.query(Recommendation).filter(Recommendation.id == recommendation_id).first()
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    # Check if user has access to this recommendation's organization
    if current_user.role != UserRole.ADMIN:
        user_orgs = [org.id for org in current_user.organizations]
        if recommendation.organization_id not in user_orgs:
            raise HTTPException(status_code=403, detail="Not authorized to update this recommendation")

    # Update fields
    if recommendation_update.is_resolved is not None:
        recommendation.is_resolved = recommendation_update.is_resolved
        if recommendation.is_resolved:
            recommendation.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(recommendation)
    return recommendation


# Chat Assistant Endpoints
@app.post("/chat", response_model=ChatResponse, tags=["Assistant"])
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Chat with the ML assistant.
    """
    # If organization_id not provided in request, use user's context
    if not request.organization_id and hasattr(current_user, "organization_id"):
        request.organization_id = current_user.organization_id

    response = chat_with_assistant(db=db, request=request)
    return response


# Organization Management Endpoints
@app.post("/organizations", response_model=OrganizationResponse)
def create_organization(
    organization: OrganizationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new organization. Any authenticated user can create an organization.
    """
    # Create new organization
    db_organization = Organization(
        name=organization.name,
        description=organization.description,
        subscription_plan=organization.subscription_plan,
        billing_email=organization.billing_email or current_user.email,
    )

    db.add(db_organization)
    db.commit()
    db.refresh(db_organization)

    # Add the creator as an admin of the organization
    org_role = OrganizationRole(
        user_id=current_user.id,
        organization_id=db_organization.id,
        role=UserRole.ADMIN
    )

    db.add(org_role)

    # Add the user to the organization's users
    db_organization.users.append(current_user)

    db.commit()

    return db_organization


@app.get("/organizations", response_model=List[OrganizationResponse])
def read_organizations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all organizations that the current user is a member of.
    """
    # Get organizations where the user is a member
    organizations = (
        db.query(Organization)
        .join(user_organization)
        .filter(user_organization.c.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return organizations


@app.get("/organizations/{organization_id}", response_model=OrganizationResponse)
def read_organization(
    organization_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Get details of a specific organization. User must be a member of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    organization = db.query(Organization).filter(Organization.id == organization_id).first()

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    return organization


@app.put("/organizations/{organization_id}", response_model=OrganizationResponse)
def update_organization(
    organization_id: int,
    organization_update: OrganizationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update an organization. User must be an admin of the organization.
    """
    # Get the organization
    organization = db.query(Organization).filter(Organization.id == organization_id).first()

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Check if the user is an admin of the organization
    org_role = db.query(OrganizationRole).filter(
        OrganizationRole.user_id == current_user.id,
        OrganizationRole.organization_id == organization_id,
    ).first()

    if not org_role or org_role.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can update the organization",
        )

    # Update organization fields
    if organization_update.name is not None:
        organization.name = organization_update.name

    if organization_update.description is not None:
        organization.description = organization_update.description

    if organization_update.subscription_plan is not None:
        organization.subscription_plan = organization_update.subscription_plan

    if organization_update.billing_email is not None:
        organization.billing_email = organization_update.billing_email

    if organization_update.is_active is not None:
        organization.is_active = organization_update.is_active

    db.commit()
    db.refresh(organization)

    return organization


@app.delete("/organizations/{organization_id}", response_model=Dict[str, Any])
def delete_organization(
    organization_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete an organization. User must be an admin of the organization.
    """
    # Get the organization
    organization = db.query(Organization).filter(Organization.id == organization_id).first()

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Check if the user is an admin of the organization
    org_role = db.query(OrganizationRole).filter(
        OrganizationRole.user_id == current_user.id,
        OrganizationRole.organization_id == organization_id,
    ).first()

    if not org_role or org_role.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization admins can delete the organization",
        )

    # Delete the organization
    db.delete(organization)
    db.commit()

    return {"message": f"Organization {organization_id} deleted successfully"}


# Project Management Endpoints
@app.post("/organizations/{organization_id}/projects", response_model=ProjectResponse)
def create_project(
    organization_id: int,
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Create a new project within an organization. User must be a member of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin or analyst role in the organization
    if current_user.current_org_role not in [UserRole.ADMIN, UserRole.ANALYST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and analysts can create projects",
        )

    # Create new project
    db_project = Project(
        name=project.name,
        description=project.description,
        organization_id=organization_id,
    )

    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    return db_project


@app.get("/organizations/{organization_id}/projects", response_model=List[ProjectResponse])
def read_projects(
    organization_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Get all projects within an organization. User must be a member of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Get projects for the organization
    projects = (
        db.query(Project)
        .filter(Project.organization_id == organization_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return projects


@app.get("/organizations/{organization_id}/projects/{project_id}", response_model=ProjectResponse)
def read_project(
    organization_id: int,
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Get details of a specific project. User must be a member of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Get the project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.organization_id == organization_id
    ).first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return project


@app.put("/organizations/{organization_id}/projects/{project_id}", response_model=ProjectResponse)
def update_project(
    organization_id: int,
    project_id: int,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Update a project. User must be an admin or analyst in the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin or analyst role in the organization
    if current_user.current_org_role not in [UserRole.ADMIN, UserRole.ANALYST]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and analysts can update projects",
        )

    # Get the project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.organization_id == organization_id
    ).first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Update project fields
    if project_update.name is not None:
        project.name = project_update.name

    if project_update.description is not None:
        project.description = project_update.description

    db.commit()
    db.refresh(project)

    return project


@app.delete("/organizations/{organization_id}/projects/{project_id}", response_model=Dict[str, Any])
def delete_project(
    organization_id: int,
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Delete a project. User must be an admin in the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete projects",
        )

    # Get the project
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.organization_id == organization_id
    ).first()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Delete the project
    db.delete(project)
    db.commit()

    return {"message": f"Project {project_id} deleted successfully"}


# Team Management Endpoints
@app.get("/organizations/{organization_id}/members", response_model=List[UserResponse])
def read_organization_members(
    organization_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Get all members of an organization. User must be a member of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Get users who are members of the organization
    users = (
        db.query(User)
        .join(user_organization)
        .filter(user_organization.c.organization_id == organization_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return users


@app.post("/organizations/{organization_id}/members", response_model=OrganizationRoleResponse)
def add_organization_member(
    organization_id: int,
    member: OrganizationRoleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Add a user to an organization with a specific role. User must be an admin of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can add members to the organization",
        )

    # Check if the user exists
    user = db.query(User).filter(User.id == member.user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Check if the user is already a member of the organization
    existing_role = db.query(OrganizationRole).filter(
        OrganizationRole.user_id == member.user_id,
        OrganizationRole.organization_id == organization_id
    ).first()

    if existing_role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of the organization",
        )

    # Add the user to the organization with the specified role
    org_role = OrganizationRole(
        user_id=member.user_id,
        organization_id=organization_id,
        role=member.role
    )

    db.add(org_role)

    # Add the user to the organization's users
    organization = db.query(Organization).filter(Organization.id == organization_id).first()
    organization.users.append(user)

    db.commit()
    db.refresh(org_role)

    return org_role


@app.put("/organizations/{organization_id}/members/{user_id}", response_model=OrganizationRoleResponse)
def update_organization_member_role(
    organization_id: int,
    user_id: int,
    role_update: OrganizationRoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Update a user's role in an organization. User must be an admin of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can update member roles",
        )

    # Check if the user is a member of the organization
    org_role = db.query(OrganizationRole).filter(
        OrganizationRole.user_id == user_id,
        OrganizationRole.organization_id == organization_id
    ).first()

    if org_role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of the organization",
        )

    # Prevent removing the last admin
    if org_role.role == UserRole.ADMIN and role_update.role != UserRole.ADMIN:
        # Count how many admins are in the organization
        admin_count = db.query(OrganizationRole).filter(
            OrganizationRole.organization_id == organization_id,
            OrganizationRole.role == UserRole.ADMIN
        ).count()

        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last admin from the organization",
            )

    # Update the user's role
    org_role.role = role_update.role
    db.commit()
    db.refresh(org_role)

    return org_role


@app.delete("/organizations/{organization_id}/members/{user_id}", response_model=Dict[str, Any])
def remove_organization_member(
    organization_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Remove a user from an organization. User must be an admin of the organization.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can remove members from the organization",
        )

    # Check if the user is a member of the organization
    org_role = db.query(OrganizationRole).filter(
        OrganizationRole.user_id == user_id,
        OrganizationRole.organization_id == organization_id
    ).first()

    if org_role is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of the organization",
        )

    # Prevent removing the last admin
    if org_role.role == UserRole.ADMIN:
        # Count how many admins are in the organization
        admin_count = db.query(OrganizationRole).filter(
            OrganizationRole.organization_id == organization_id,
            OrganizationRole.role == UserRole.ADMIN
        ).count()

        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove the last admin from the organization",
            )

    # Remove the user from the organization
    organization = db.query(Organization).filter(Organization.id == organization_id).first()
    user = db.query(User).filter(User.id == user_id).first()

    if user in organization.users:
        organization.users.remove(user)

    # Delete the organization role
    db.delete(org_role)
    db.commit()

    return {"message": f"User {user_id} removed from organization {organization_id} successfully"}


# SaaS Billing Endpoints (Placeholders for future integration)
@app.get("/organizations/{organization_id}/subscription", response_model=Dict[str, Any])
def get_organization_subscription(
    organization_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Get the subscription details for an organization. User must be a member of the organization.
    This is a placeholder for future integration with billing systems like Stripe or Paddle.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Get the organization
    organization = db.query(Organization).filter(Organization.id == organization_id).first()

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # In a real implementation, this would fetch subscription details from a billing system
    # For now, return placeholder data based on the organization's subscription_plan field
    subscription_details = {
        "plan": organization.subscription_plan,
        "status": "active",
        "billing_email": organization.billing_email,
        "next_billing_date": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "features": get_plan_features(organization.subscription_plan),
        "usage": {
            "models": db.query(InferenceLog).filter(
                InferenceLog.organization_id == organization_id
            ).distinct(InferenceLog.model_name).count(),
            "projects": db.query(Project).filter(
                Project.organization_id == organization_id
            ).count(),
            "team_members": db.query(OrganizationRole).filter(
                OrganizationRole.organization_id == organization_id
            ).count(),
        }
    }

    return subscription_details


@app.put("/organizations/{organization_id}/subscription", response_model=Dict[str, Any])
def update_organization_subscription(
    organization_id: int,
    subscription_plan: str = Query(..., description="Subscription plan (Basic, Pro, Enterprise)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Update the subscription plan for an organization. User must be an admin of the organization.
    This is a placeholder for future integration with billing systems like Stripe or Paddle.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can update the subscription plan",
        )

    # Validate the subscription plan
    if subscription_plan not in ["Basic", "Pro", "Enterprise"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subscription plan. Must be one of: Basic, Pro, Enterprise",
        )

    # Get the organization
    organization = db.query(Organization).filter(Organization.id == organization_id).first()

    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Update the subscription plan
    organization.subscription_plan = subscription_plan
    db.commit()

    # In a real implementation, this would update the subscription in a billing system
    # For now, return placeholder data
    return {
        "message": f"Subscription plan updated to {subscription_plan}",
        "plan": subscription_plan,
        "features": get_plan_features(subscription_plan),
    }


@app.post("/organizations/{organization_id}/billing/payment-method", response_model=Dict[str, Any])
def add_payment_method(
    organization_id: int,
    payment_token: str = Query(..., description="Payment method token from payment processor"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_organization_member),
):
    """
    Add a payment method to an organization. User must be an admin of the organization.
    This is a placeholder for future integration with billing systems like Stripe or Paddle.
    """
    # Check if the user's current organization context matches the requested organization
    if current_user.current_organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not a member of this organization",
        )

    # Check if the user has admin role in the organization
    if current_user.current_org_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can add payment methods",
        )

    # In a real implementation, this would add the payment method to a billing system
    # For now, return placeholder data
    return {
        "message": "Payment method added successfully",
        "payment_method": {
            "id": "pm_" + payment_token[:8],
            "type": "credit_card",
            "last4": payment_token[-4:],
            "exp_month": 12,
            "exp_year": 2025,
        }
    }


# Helper function for subscription plans
def get_plan_features(plan: str) -> Dict[str, Any]:
    """
    Get the features for a subscription plan.
    """
    plans = {
        "Basic": {
            "price": 49,
            "models": 5,
            "projects": 3,
            "team_members": 5,
            "log_retention_days": 30,
            "support": "email",
        },
        "Pro": {
            "price": 99,
            "models": 20,
            "projects": 10,
            "team_members": 15,
            "log_retention_days": 90,
            "support": "priority_email",
            "custom_alerts": True,
        },
        "Enterprise": {
            "price": 299,
            "models": "unlimited",
            "projects": "unlimited",
            "team_members": "unlimited",
            "log_retention_days": 365,
            "support": "dedicated",
            "custom_alerts": True,
            "sla": "99.9%",
            "custom_integrations": True,
        }
    }

    return plans.get(plan, plans["Basic"])
