from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import TypeDecorator
import json
from datetime import datetime

from .database import Base

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
