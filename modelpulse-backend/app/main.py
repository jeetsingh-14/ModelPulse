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

from .models import (
    InferenceLog,
    AlertThreshold,
    DriftMetrics,
    DriftSeverity,
    User,
    UserRole,
)
from .drift import compute_drift_metrics, get_drift_summary
from .auth import (
    create_access_token,
    get_current_user,
    get_current_active_user,
    get_admin_user,
    get_analyst_user,
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
)
from .database import get_db, engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

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


@app.get("/")
def read_root():
    return {"message": "Hello from ModelPulse"}


# Authentication Endpoints
@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
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

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires,
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_analyst_user),  # Require analyst or admin role
):
    db_log = InferenceLog(
        model_name=log.model_name,
        timestamp=log.timestamp,
        input_shape=log.input_shape,
        latency_ms=log.latency_ms,
        confidence=log.confidence,
        output_class=log.output_class,
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)

    # Check for threshold breaches
    alerts = []

    # Get active thresholds that apply to this model or all models
    thresholds = (
        db.query(AlertThreshold)
        .filter(
            (AlertThreshold.model_name == log.model_name)
            | (AlertThreshold.model_name == None)
        )
        .filter(AlertThreshold.is_active == True)
        .all()
    )

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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    """
    Get inference logs with optional filtering by model, output class, and date range.
    """
    query = db.query(InferenceLog)

    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)

    if output_class:
        query = query.filter(InferenceLog.output_class == output_class)

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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    format: str = "csv",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # Any authenticated user
):
    """
    Export inference logs to CSV format with optional filtering.
    """
    # Build query with filters
    query = db.query(InferenceLog)

    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)

    if output_class:
        query = query.filter(InferenceLog.output_class == output_class)

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
