from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

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