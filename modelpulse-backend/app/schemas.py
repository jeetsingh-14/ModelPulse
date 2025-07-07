from pydantic import BaseModel, Field, EmailStr, validator, HttpUrl
from typing import List, Optional, Literal, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .models import (
    UserRole, 
    DriftSeverity, 
    IntegrationType, 
    ExportType, 
    WebhookEvent, 
    AutomationPolicyType, 
    AutomationAction, 
    ModelValidationStatus, 
    JobStatus,
    AuditActionType
)


# Organization Schemas
class OrganizationBase(BaseModel):
    name: str
    description: Optional[str] = None
    subscription_plan: str = "Basic"
    billing_email: Optional[str] = None
    is_active: bool = True

    # Compliance fields
    data_retention_days: int = 365
    pii_handling_policy: str = "encrypt"
    data_residency_region: Optional[str] = None
    gdpr_compliant: bool = False
    hipaa_compliant: bool = False
    soc2_compliant: bool = False
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    subscription_plan: Optional[str] = None
    billing_email: Optional[str] = None
    is_active: Optional[bool] = None

    # Compliance fields
    data_retention_days: Optional[int] = None
    pii_handling_policy: Optional[str] = None
    data_residency_region: Optional[str] = None
    gdpr_compliant: Optional[bool] = None
    hipaa_compliant: Optional[bool] = None
    soc2_compliant: Optional[bool] = None
    encryption_at_rest: Optional[bool] = None
    encryption_in_transit: Optional[bool] = None


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

    # SSO fields
    identity_provider: Optional[str] = None
    external_id: Optional[str] = None

    # MFA fields
    mfa_enabled: bool = False
    mfa_type: Optional[str] = None


class UserCreate(UserBase):
    password: Optional[str] = None

    @validator("password")
    def password_strength(cls, v):
        # Skip validation if password is None (for SSO users)
        if v is None:
            return v

        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @validator("password", "identity_provider")
    def validate_auth_method(cls, v, values):
        # Ensure either password or identity_provider is provided
        if "identity_provider" in values and values["identity_provider"] is not None:
            # SSO user, password can be None
            return v
        elif "password" in values and values["password"] is not None:
            # Password user, identity_provider can be None
            return v
        else:
            # Neither password nor identity_provider provided
            raise ValueError("Either password or identity_provider must be provided")


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

    # SSO fields
    identity_provider: Optional[str] = None
    external_id: Optional[str] = None

    # MFA fields
    mfa_enabled: Optional[bool] = None
    mfa_type: Optional[str] = None
    mfa_secret: Optional[str] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    # Audit fields
    last_login: Optional[datetime] = None
    login_count: Optional[int] = None

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


# Integration Schemas
class IntegrationBase(BaseModel):
    name: str
    integration_type: IntegrationType
    config: Dict[str, Any]
    is_active: bool = True
    organization_id: int
    project_id: Optional[int] = None


class IntegrationCreate(IntegrationBase):
    pass


class IntegrationUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class IntegrationResponse(IntegrationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Export Configuration Schemas
class ExportConfigurationBase(BaseModel):
    name: str
    export_type: ExportType
    config: Dict[str, Any]
    schedule: Optional[str] = None
    is_active: bool = True
    organization_id: int
    project_id: Optional[int] = None


class ExportConfigurationCreate(ExportConfigurationBase):
    pass


class ExportConfigurationUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    schedule: Optional[str] = None
    is_active: Optional[bool] = None


class ExportConfigurationResponse(ExportConfigurationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Webhook Schemas
class WebhookBase(BaseModel):
    name: str
    url: HttpUrl
    events: List[WebhookEvent]
    headers: Optional[Dict[str, str]] = None
    is_active: bool = True
    organization_id: int
    project_id: Optional[int] = None


class WebhookCreate(WebhookBase):
    pass


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    events: Optional[List[WebhookEvent]] = None
    headers: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = None


class WebhookHistoryResponse(BaseModel):
    id: int
    webhook_id: int
    event: WebhookEvent
    payload: Dict[str, Any]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True


class WebhookResponse(WebhookBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Automation Policy Schemas
class AutomationPolicyBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_name: Optional[str] = None
    policy_type: AutomationPolicyType
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    is_active: bool = True


class AutomationPolicyCreate(AutomationPolicyBase):
    pass


class AutomationPolicyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = None
    policy_type: Optional[AutomationPolicyType] = None
    conditions: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    is_active: Optional[bool] = None


class AutomationPolicyResponse(AutomationPolicyBase):
    id: int
    organization_id: int
    project_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# Retraining Job Schemas
class RetrainingJobBase(BaseModel):
    model_name: str
    training_params: Optional[Dict[str, Any]] = None


class RetrainingJobCreate(RetrainingJobBase):
    policy_id: Optional[int] = None
    integration_id: Optional[int] = None


class RetrainingJobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    training_params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None


class RetrainingJobResponse(RetrainingJobBase):
    id: int
    status: JobStatus
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    policy_id: Optional[int] = None
    integration_id: Optional[int] = None
    organization_id: int
    project_id: Optional[int] = None

    class Config:
        orm_mode = True


# Model Validation Schemas
class ModelValidationBase(BaseModel):
    model_name: str
    model_version: str
    validation_params: Optional[Dict[str, Any]] = None


class ModelValidationCreate(ModelValidationBase):
    retraining_job_id: Optional[int] = None


class ModelValidationUpdate(BaseModel):
    status: Optional[ModelValidationStatus] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None


class ModelValidationResponse(ModelValidationBase):
    id: int
    status: ModelValidationStatus
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    retraining_job_id: Optional[int] = None
    organization_id: int
    project_id: Optional[int] = None

    class Config:
        orm_mode = True


# Model Deployment Schemas
class ModelDeploymentBase(BaseModel):
    model_name: str
    model_version: str
    deployment_params: Optional[Dict[str, Any]] = None


class ModelDeploymentCreate(ModelDeploymentBase):
    retraining_job_id: Optional[int] = None
    validation_id: Optional[int] = None
    integration_id: Optional[int] = None
    previous_deployment_id: Optional[int] = None
    is_rollback: bool = False


class ModelDeploymentUpdate(BaseModel):
    status: Optional[JobStatus] = None
    deployment_params: Optional[Dict[str, Any]] = None
    endpoint_url: Optional[str] = None
    logs: Optional[str] = None


class ModelDeploymentResponse(ModelDeploymentBase):
    id: int
    status: JobStatus
    endpoint_url: Optional[str] = None
    logs: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    retraining_job_id: Optional[int] = None
    validation_id: Optional[int] = None
    integration_id: Optional[int] = None
    organization_id: int
    project_id: Optional[int] = None
    previous_deployment_id: Optional[int] = None
    is_rollback: bool

    class Config:
        orm_mode = True


# Audit Log Schemas
class AuditLogBase(BaseModel):
    action: AuditActionType
    resource_type: str
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogCreate(AuditLogBase):
    user_id: Optional[int] = None
    organization_id: Optional[int] = None
    project_id: Optional[int] = None


class AuditLogResponse(AuditLogBase):
    id: int
    timestamp: datetime
    user_id: Optional[int] = None
    organization_id: Optional[int] = None
    project_id: Optional[int] = None

    class Config:
        orm_mode = True


class AuditLogFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    action: Optional[AuditActionType] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    user_id: Optional[int] = None
    status: Optional[str] = None
