from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import relationship
import json
from datetime import datetime
from enum import Enum
from passlib.context import CryptContext

from .database import Base

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class ArrayType(TypeDecorator):
    impl = String

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


# Association table for many-to-many relationship between users and organizations
user_organization = Table(
    "user_organization",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("organization_id", Integer, ForeignKey("organizations.id"), primary_key=True),
)


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    subscription_plan = Column(String, default="Basic")  # Basic, Pro, Enterprise
    billing_email = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Compliance fields
    data_retention_days = Column(Integer, default=365)  # Default to 1 year
    pii_handling_policy = Column(String, default="encrypt")  # encrypt, anonymize, none
    data_residency_region = Column(String, nullable=True)  # AWS region, GCP region, etc.
    gdpr_compliant = Column(Boolean, default=False)
    hipaa_compliant = Column(Boolean, default=False)
    soc2_compliant = Column(Boolean, default=False)
    encryption_at_rest = Column(Boolean, default=True)
    encryption_in_transit = Column(Boolean, default=True)

    # Relationships
    users = relationship("User", secondary=user_organization, back_populates="organizations")
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    inference_logs = relationship("InferenceLog", back_populates="organization", cascade="all, delete-orphan")
    alert_thresholds = relationship("AlertThreshold", back_populates="organization", cascade="all, delete-orphan")
    drift_metrics = relationship("DriftMetrics", back_populates="organization", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="organization", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="organization", cascade="all, delete-orphan")
    export_configs = relationship("ExportConfiguration", back_populates="organization", cascade="all, delete-orphan")
    webhooks = relationship("Webhook", back_populates="organization", cascade="all, delete-orphan")
    automation_policies = relationship("AutomationPolicy", back_populates="organization", cascade="all, delete-orphan")
    retraining_jobs = relationship("RetrainingJob", back_populates="organization", cascade="all, delete-orphan")
    model_validations = relationship("ModelValidation", back_populates="organization", cascade="all, delete-orphan")
    model_deployments = relationship("ModelDeployment", back_populates="organization", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    inference_logs = relationship("InferenceLog", back_populates="project", cascade="all, delete-orphan")
    alert_thresholds = relationship("AlertThreshold", back_populates="project", cascade="all, delete-orphan")
    drift_metrics = relationship("DriftMetrics", back_populates="project", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="project", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="project", cascade="all, delete-orphan")
    export_configs = relationship("ExportConfiguration", back_populates="project", cascade="all, delete-orphan")
    webhooks = relationship("Webhook", back_populates="project", cascade="all, delete-orphan")
    automation_policies = relationship("AutomationPolicy", back_populates="project", cascade="all, delete-orphan")
    retraining_jobs = relationship("RetrainingJob", back_populates="project", cascade="all, delete-orphan")
    model_validations = relationship("ModelValidation", back_populates="project", cascade="all, delete-orphan")
    model_deployments = relationship("ModelDeployment", back_populates="project", cascade="all, delete-orphan")


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_shape = Column(ArrayType)  # Stored as JSON string for SQLite compatibility
    latency_ms = Column(Float)
    confidence = Column(Float)
    output_class = Column(String, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="inference_logs")
    project = relationship("Project", back_populates="inference_logs")


class AlertThreshold(Base):
    __tablename__ = "alert_thresholds"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(
        String, index=True, nullable=True
    )  # If null, applies to all models
    metric_name = Column(String, index=True)  # 'latency_ms', 'confidence'
    threshold_value = Column(Float)
    is_active = Column(Boolean, default=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="alert_thresholds")
    project = relationship("Project", back_populates="alert_thresholds")


class DriftSeverity(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AuditActionType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    ACCESS = "access"
    CONFIG_CHANGE = "config_change"


class RecommendationAction(str, Enum):
    RETRAIN = "retrain"
    MONITOR = "monitor"
    INVESTIGATE = "investigate"
    ROLLBACK = "rollback"
    OPTIMIZE = "optimize"
    DEPLOY = "deploy"
    VALIDATE = "validate"


class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class IntegrationType(str, Enum):
    MLFLOW = "mlflow"
    SAGEMAKER = "sagemaker"
    VERTEX_AI = "vertex_ai"


class ExportType(str, Enum):
    ELK = "elk"
    SPLUNK = "splunk"
    DATADOG = "datadog"


class WebhookEvent(str, Enum):
    DRIFT_DETECTED = "drift_detected"
    THRESHOLD_BREACHED = "threshold_breached"
    RECOMMENDATION_CREATED = "recommendation_created"
    MODEL_REGISTERED = "model_registered"
    RETRAINING_STARTED = "retraining_started"
    RETRAINING_COMPLETED = "retraining_completed"
    MODEL_DEPLOYED = "model_deployed"
    VALIDATION_FAILED = "validation_failed"


class OrganizationRole(Base):
    __tablename__ = "organization_roles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    role = Column(String, default=UserRole.VIEWER)

    # Relationships
    user = relationship("User", back_populates="organization_roles")
    organization = relationship("Organization")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)  # Nullable for SSO users
    full_name = Column(String, nullable=True)
    role = Column(String, default=UserRole.VIEWER)  # Global role (for backward compatibility)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # SSO fields
    identity_provider = Column(String, nullable=True)  # "google", "okta", "azure", etc.
    external_id = Column(String, nullable=True, index=True)  # ID from the identity provider

    # MFA fields
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String, nullable=True)
    mfa_type = Column(String, nullable=True)  # "totp", "sms", etc.

    # Audit fields
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)
    last_ip_address = Column(String, nullable=True)
    last_user_agent = Column(String, nullable=True)

    # Relationships
    organizations = relationship("Organization", secondary=user_organization, back_populates="users")
    organization_roles = relationship("OrganizationRole", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password: str) -> bool:
        """Verify the provided password against the stored hash"""
        return pwd_context.verify(password, self.hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate a password hash"""
        return pwd_context.hash(password)

    def get_role_in_organization(self, organization_id: int) -> str:
        """Get the user's role in a specific organization"""
        for org_role in self.organization_roles:
            if org_role.organization_id == organization_id:
                return org_role.role
        return None


class DriftMetrics(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Input drift metrics
    input_kl_divergence = Column(Float)
    input_psi = Column(Float)  # Population Stability Index
    input_distribution_reference = Column(JSON, nullable=True)
    input_distribution_current = Column(JSON, nullable=True)

    # Output drift metrics
    output_kl_divergence = Column(Float)
    output_psi = Column(Float)
    output_distribution_reference = Column(JSON, nullable=True)
    output_distribution_current = Column(JSON, nullable=True)

    # Overall drift status
    drift_severity = Column(String, default=DriftSeverity.OK)

    # Explanation
    explanation = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="drift_metrics")
    project = relationship("Project", back_populates="drift_metrics")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    action = Column(String, index=True)  # RecommendationAction enum as string
    priority = Column(String, index=True)  # RecommendationPriority enum as string
    description = Column(String)
    reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="recommendations")
    project = relationship("Project", back_populates="recommendations")


class Integration(Base):
    __tablename__ = "integrations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    integration_type = Column(String, index=True)  # IntegrationType enum as string
    config = Column(JSON, nullable=False)  # Stores connection details, credentials, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="integrations")
    project = relationship("Project", back_populates="integrations")


class ExportConfiguration(Base):
    __tablename__ = "export_configurations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    export_type = Column(String, index=True)  # ExportType enum as string
    config = Column(JSON, nullable=False)  # Stores connection details, credentials, etc.
    schedule = Column(String, nullable=True)  # Cron expression for scheduled exports
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="export_configs")
    project = relationship("Project", back_populates="export_configs")


class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    url = Column(String, nullable=False)
    events = Column(JSON, nullable=False)  # List of WebhookEvent enum values
    headers = Column(JSON, nullable=True)  # Custom headers to include
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="webhooks")
    project = relationship("Project", back_populates="webhooks")
    history = relationship("WebhookHistory", back_populates="webhook", cascade="all, delete-orphan")


class WebhookHistory(Base):
    __tablename__ = "webhook_history"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("webhooks.id"))
    event = Column(String, index=True)  # WebhookEvent enum as string
    payload = Column(JSON, nullable=False)
    response_status = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    webhook = relationship("Webhook", back_populates="history")


class AutomationPolicyType(str, Enum):
    DRIFT = "drift"
    PERFORMANCE = "performance"
    SCHEDULE = "schedule"


class AutomationAction(str, Enum):
    RETRAIN = "retrain"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    NOTIFY = "notify"


class ModelValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AutomationPolicy(Base):
    __tablename__ = "automation_policies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    model_name = Column(String, index=True, nullable=True)  # If null, applies to all models
    policy_type = Column(String, index=True)  # AutomationPolicyType enum as string

    # Trigger conditions
    conditions = Column(JSON, nullable=False)  # E.g., {"drift_severity": "CRITICAL", "metric": "latency_ms", "threshold": 100}

    # Actions to take when conditions are met
    actions = Column(JSON, nullable=False)  # List of AutomationAction enum values with parameters

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    organization = relationship("Organization")
    project = relationship("Project")
    retraining_jobs = relationship("RetrainingJob", back_populates="policy", cascade="all, delete-orphan")


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    status = Column(String, index=True)  # JobStatus enum as string

    # Training parameters
    training_params = Column(JSON, nullable=True)  # Hyperparameters, dataset config, etc.

    # Results
    metrics = Column(JSON, nullable=True)  # Training and validation metrics

    # Logs
    logs = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # References
    policy_id = Column(Integer, ForeignKey("automation_policies.id"), nullable=True)
    integration_id = Column(Integer, ForeignKey("integrations.id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    policy = relationship("AutomationPolicy", back_populates="retraining_jobs")
    integration = relationship("Integration")
    organization = relationship("Organization")
    project = relationship("Project")
    model_validations = relationship("ModelValidation", back_populates="retraining_job", cascade="all, delete-orphan")
    model_deployments = relationship("ModelDeployment", back_populates="retraining_job", cascade="all, delete-orphan")


class ModelValidation(Base):
    __tablename__ = "model_validations"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    model_version = Column(String, index=True)
    status = Column(String, index=True)  # ModelValidationStatus enum as string

    # Validation parameters
    validation_params = Column(JSON, nullable=True)  # Test dataset, metrics to evaluate, etc.

    # Results
    metrics = Column(JSON, nullable=True)  # Validation metrics

    # Logs
    logs = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # References
    retraining_job_id = Column(Integer, ForeignKey("retraining_jobs.id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    retraining_job = relationship("RetrainingJob", back_populates="model_validations")
    organization = relationship("Organization")
    project = relationship("Project")


class ModelDeployment(Base):
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    model_version = Column(String, index=True)
    status = Column(String, index=True)  # JobStatus enum as string

    # Deployment parameters
    deployment_params = Column(JSON, nullable=True)  # Endpoint config, scaling, etc.

    # Results
    endpoint_url = Column(String, nullable=True)

    # Logs
    logs = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # References
    retraining_job_id = Column(Integer, ForeignKey("retraining_jobs.id"), nullable=True)
    validation_id = Column(Integer, ForeignKey("model_validations.id"), nullable=True)
    integration_id = Column(Integer, ForeignKey("integrations.id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # For rollback tracking
    previous_deployment_id = Column(Integer, ForeignKey("model_deployments.id"), nullable=True)
    is_rollback = Column(Boolean, default=False)

    # Relationships
    retraining_job = relationship("RetrainingJob", back_populates="model_deployments")
    validation = relationship("ModelValidation")
    integration = relationship("Integration")
    organization = relationship("Organization")
    project = relationship("Project")
    previous_deployment = relationship("ModelDeployment", remote_side=[id], uselist=False)
    next_deployments = relationship("ModelDeployment", backref="next_deployment")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action = Column(String, index=True)  # AuditActionType enum as string
    resource_type = Column(String, index=True)  # The type of resource being acted upon (e.g., "user", "model", "organization")
    resource_id = Column(String, index=True, nullable=True)  # The ID of the resource being acted upon
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # The user who performed the action
    ip_address = Column(String, nullable=True)  # The IP address of the user
    user_agent = Column(String, nullable=True)  # The user agent of the user
    details = Column(JSON, nullable=True)  # Additional details about the action
    status = Column(String, nullable=True)  # Success or failure status
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)

    # Relationships
    user = relationship("User")
    organization = relationship("Organization")
    project = relationship("Project")
