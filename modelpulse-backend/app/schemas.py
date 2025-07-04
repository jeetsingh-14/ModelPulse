from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .models import UserRole


class InferenceLogBase(BaseModel):
    model_name: str
    timestamp: datetime
    input_shape: List[int]
    latency_ms: float
    confidence: float
    output_class: str


class InferenceLogCreate(InferenceLogBase):
    pass


class InferenceLogResponse(InferenceLogBase):
    id: int

    class Config:
        orm_mode = True


class AlertThresholdBase(BaseModel):
    model_name: Optional[str] = None
    metric_name: Literal["latency_ms", "confidence"]
    threshold_value: float
    is_active: bool = True


class AlertThresholdCreate(AlertThresholdBase):
    pass


class AlertThresholdUpdate(BaseModel):
    model_name: Optional[str] = None
    metric_name: Optional[Literal["latency_ms", "confidence"]] = None
    threshold_value: Optional[float] = None
    is_active: Optional[bool] = None


class AlertThresholdResponse(AlertThresholdBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Alert(BaseModel):
    model_name: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: datetime


class AnalyticsSummary(BaseModel):
    avg_latency: float
    avg_confidence: float
    total_requests: int
    requests_per_minute: float
    model_distribution: Dict[str, int]
    class_distribution: Dict[str, int]
    latency_over_time: List[Dict[str, Any]]
    confidence_distribution: List[Dict[str, Any]]


class DriftSeverity(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriftMetricsBase(BaseModel):
    model_name: str
    input_kl_divergence: float
    input_psi: float
    input_distribution_reference: Optional[Dict[str, Any]] = None
    input_distribution_current: Optional[Dict[str, Any]] = None
    output_kl_divergence: float
    output_psi: float
    output_distribution_reference: Optional[Dict[str, Any]] = None
    output_distribution_current: Optional[Dict[str, Any]] = None
    drift_severity: DriftSeverity = DriftSeverity.OK
    explanation: Optional[str] = None


class DriftMetricsCreate(DriftMetricsBase):
    timestamp: Optional[datetime] = None


class DriftMetricsResponse(DriftMetricsBase):
    id: int
    timestamp: datetime
    created_at: datetime

    class Config:
        orm_mode = True


class DriftSummary(BaseModel):
    model_name: str
    current_severity: DriftSeverity
    input_drift_trend: List[Dict[str, Any]]
    output_drift_trend: List[Dict[str, Any]]
    latest_explanation: Optional[str] = None


# User Authentication Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER


class UserCreate(UserBase):
    password: str

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
