from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .models import UserRole, DriftSeverity


# Organization Schemas
class OrganizationBase(BaseModel):
    name: str
    description: Optional[str] = None
    subscription_plan: str = "Basic"
    billing_email: Optional[str] = None
    is_active: bool = True


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    subscription_plan: Optional[str] = None
    billing_email: Optional[str] = None
    is_active: Optional[bool] = None


class OrganizationResponse(OrganizationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Project Schemas
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    organization_id: int


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Organization Role Schemas
class OrganizationRoleBase(BaseModel):
    user_id: int
    organization_id: int
    role: UserRole = UserRole.VIEWER


class OrganizationRoleCreate(OrganizationRoleBase):
    pass


class OrganizationRoleUpdate(BaseModel):
    role: Optional[UserRole] = None


class OrganizationRoleResponse(OrganizationRoleBase):
    id: int

    class Config:
        orm_mode = True


# Organization Invitation Schemas
class OrganizationInvitationBase(BaseModel):
    email: EmailStr
    organization_id: int
    role: UserRole = UserRole.VIEWER


class OrganizationInvitationCreate(OrganizationInvitationBase):
    pass


class OrganizationInvitationResponse(OrganizationInvitationBase):
    id: int
    created_at: datetime
    accepted: bool = False

    class Config:
        orm_mode = True


class InferenceLogBase(BaseModel):
    model_name: str
    timestamp: datetime
    input_shape: List[int]
    latency_ms: float
    confidence: float
    output_class: str
    organization_id: int
    project_id: Optional[int] = None


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
    organization_id: int
    project_id: Optional[int] = None


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
    organization_id: int
    project_id: Optional[int] = None


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
    role: UserRole = UserRole.VIEWER  # Global role (for backward compatibility)


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
    organizations: Optional[List[OrganizationResponse]] = None

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    organization_id: Optional[int] = None
    org_role: Optional[UserRole] = None


# Predictive Alerts Schemas
class ForecastPoint(BaseModel):
    timestamp: str
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    predicted_severity: Optional[DriftSeverity] = None
    threshold_breach: Optional[bool] = None
    threshold_id: Optional[int] = None
    threshold_value: Optional[float] = None


class DriftPrediction(BaseModel):
    model_name: str
    input_psi_forecast: List[ForecastPoint]
    output_psi_forecast: List[ForecastPoint]
    predicted_breach_time: Optional[str] = None
    forecast_generated_at: str
    error: Optional[str] = None


class PerformancePrediction(BaseModel):
    model_name: str
    metric_type: str
    forecast: List[ForecastPoint]
    predicted_breach_time: Optional[str] = None
    forecast_generated_at: str
    error: Optional[str] = None


class PredictiveAlert(BaseModel):
    model_name: str
    drift_prediction: DriftPrediction
    latency_prediction: PerformancePrediction
    confidence_prediction: PerformancePrediction
    generated_at: str


# Recommendation Schemas
class RecommendationAction(str, Enum):
    RETRAIN = "retrain"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    ROLLBACK = "rollback"
    OPTIMIZE = "optimize"


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Recommendation(BaseModel):
    id: Optional[int] = None
    model_name: str
    action: RecommendationAction
    priority: RecommendationPriority
    description: str
    reason: str
    created_at: Optional[datetime] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    organization_id: int
    project_id: Optional[int] = None


class RecommendationCreate(BaseModel):
    model_name: str
    action: RecommendationAction
    priority: RecommendationPriority
    description: str
    reason: str
    organization_id: int
    project_id: Optional[int] = None


class RecommendationUpdate(BaseModel):
    is_resolved: Optional[bool] = None


class RecommendationResponse(Recommendation):
    class Config:
        orm_mode = True


# Chat Assistant Schemas
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    model_context: Optional[List[str]] = None
    organization_id: Optional[int] = None
    project_id: Optional[int] = None


class ChatResponse(BaseModel):
    message: str
    context_used: Optional[Dict[str, Any]] = None
    timestamp: datetime
