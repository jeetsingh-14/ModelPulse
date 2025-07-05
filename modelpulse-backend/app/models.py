from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    JSON,
    ForeignKey,
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


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_shape = Column(ArrayType)  # Stored as JSON string for SQLite compatibility
    latency_ms = Column(Float)
    confidence = Column(Float)
    output_class = Column(String, index=True)


class AlertThreshold(Base):
    __tablename__ = "alert_thresholds"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(
        String, index=True, nullable=True
    )  # If null, applies to all models
    metric_name = Column(String, index=True)  # 'latency_ms', 'confidence'
    threshold_value = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DriftSeverity(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    role = Column(String, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def verify_password(self, password: str) -> bool:
        """Verify the provided password against the stored hash"""
        return pwd_context.verify(password, self.hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate a password hash"""
        return pwd_context.hash(password)


class DriftMetrics(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

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
