from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from .models import InferenceLog
from .schemas import InferenceLogCreate, InferenceLogResponse
from .database import get_db, engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ModelPulse API", description="Real-time ML model monitoring platform")

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

@app.post("/log", response_model=InferenceLogResponse)
def log_inference(log: InferenceLogCreate, db: Session = Depends(get_db)):
    db_log = InferenceLog(
        model_name=log.model_name,
        timestamp=log.timestamp,
        input_shape=log.input_shape,
        latency_ms=log.latency_ms,
        confidence=log.confidence,
        output_class=log.output_class
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@app.get("/logs", response_model=List[InferenceLogResponse])
def get_logs(
    model_name: Optional[str] = None,
    output_class: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    query = db.query(InferenceLog)
    
    if model_name:
        query = query.filter(InferenceLog.model_name == model_name)
    
    if output_class:
        query = query.filter(InferenceLog.output_class == output_class)
    
    logs = query.offset(skip).limit(limit).all()
    return logs