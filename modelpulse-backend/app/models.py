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

    # Relationships
    users = relationship("User", secondary=user_organization, back_populates="organizations")
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    inference_logs = relationship("InferenceLog", back_populates="organization", cascade="all, delete-orphan")
    alert_thresholds = relationship("AlertThreshold", back_populates="organization", cascade="all, delete-orphan")
    drift_metrics = relationship("DriftMetrics", back_populates="organization", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="organization", cascade="all, delete-orphan")


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


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


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
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    role = Column(String, default=UserRole.VIEWER)  # Global role (for backward compatibility)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
