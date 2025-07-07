"""
Integration module for connecting ModelPulse to external ML platforms.

This module provides connectors for:
- MLflow Tracking Server
- Amazon SageMaker
- Google Vertex AI
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException
import json
import logging
from datetime import datetime

from .models import Integration, IntegrationType
from .schemas import IntegrationCreate, IntegrationUpdate

# Set up logging
logger = logging.getLogger(__name__)


class BaseConnector:
    """Base class for all ML platform connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        
    def validate_config(self) -> bool:
        """Validate the configuration."""
        raise NotImplementedError("Subclasses must implement validate_config")
        
    def connect(self) -> Any:
        """Connect to the platform."""
        raise NotImplementedError("Subclasses must implement connect")
        
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of models from the platform."""
        raise NotImplementedError("Subclasses must implement get_models")
        
    def register_model(self, model_name: str, model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model with the platform."""
        raise NotImplementedError("Subclasses must implement register_model")


class MLflowConnector(BaseConnector):
    """Connector for MLflow Tracking Server."""
    
    def validate_config(self) -> bool:
        """Validate MLflow configuration."""
        required_keys = ["tracking_uri"]
        return all(key in self.config for key in required_keys)
        
    def connect(self) -> Any:
        """Connect to MLflow Tracking Server."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.config["tracking_uri"])
            # Test connection by getting experiments
            mlflow.search_experiments()
            self.client = mlflow
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to MLflow: {str(e)}")
        
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of registered models from MLflow."""
        if not self.client:
            self.connect()
            
        try:
            models = self.client.search_registered_models()
            result = []
            for model in models:
                result.append({
                    "name": model.name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "status": v.status,
                            "stage": v.current_stage,
                            "creation_timestamp": v.creation_timestamp
                        } for v in model.latest_versions
                    ],
                    "tags": model.tags if hasattr(model, "tags") else {}
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get models from MLflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get models from MLflow: {str(e)}")
        
    def register_model(self, model_name: str, model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model with MLflow."""
        if not self.client:
            self.connect()
            
        try:
            # Check if model exists
            try:
                model = self.client.get_registered_model(model_name)
            except:
                # Create model if it doesn't exist
                model = self.client.create_registered_model(model_name)
                
            # Add tags to model
            for key, value in metadata.items():
                self.client.set_tag(model_name, key, value)
                
            return {
                "name": model_name,
                "registered_at": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Failed to register model with MLflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to register model with MLflow: {str(e)}")


class SageMakerConnector(BaseConnector):
    """Connector for Amazon SageMaker."""
    
    def validate_config(self) -> bool:
        """Validate SageMaker configuration."""
        required_keys = ["aws_access_key_id", "aws_secret_access_key", "region_name"]
        return all(key in self.config for key in required_keys)
        
    def connect(self) -> Any:
        """Connect to Amazon SageMaker."""
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=self.config["aws_access_key_id"],
                aws_secret_access_key=self.config["aws_secret_access_key"],
                region_name=self.config["region_name"]
            )
            self.client = session.client("sagemaker")
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to SageMaker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to SageMaker: {str(e)}")
        
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of models from SageMaker."""
        if not self.client:
            self.connect()
            
        try:
            response = self.client.list_models()
            result = []
            for model in response["Models"]:
                result.append({
                    "name": model["ModelName"],
                    "arn": model["ModelArn"],
                    "creation_time": model["CreationTime"].isoformat()
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get models from SageMaker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get models from SageMaker: {str(e)}")
        
    def register_model(self, model_name: str, model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model with SageMaker."""
        if not self.client:
            self.connect()
            
        try:
            # In SageMaker, we would typically create a model package
            # This is a simplified version
            response = self.client.create_model(
                ModelName=f"{model_name}-{model_version}",
                PrimaryContainer={
                    "Image": metadata.get("image_uri", ""),
                    "ModelDataUrl": metadata.get("model_data_url", ""),
                    "Environment": metadata.get("environment", {})
                },
                Tags=[{"Key": k, "Value": str(v)} for k, v in metadata.items() if k not in ["image_uri", "model_data_url", "environment"]]
            )
            
            return {
                "name": model_name,
                "version": model_version,
                "arn": response["ModelArn"],
                "registered_at": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Failed to register model with SageMaker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to register model with SageMaker: {str(e)}")


class VertexAIConnector(BaseConnector):
    """Connector for Google Vertex AI."""
    
    def validate_config(self) -> bool:
        """Validate Vertex AI configuration."""
        required_keys = ["project_id", "location", "credentials_json"]
        return all(key in self.config for key in required_keys)
        
    def connect(self) -> Any:
        """Connect to Google Vertex AI."""
        try:
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            import json
            
            # Parse credentials JSON
            credentials_dict = json.loads(self.config["credentials_json"])
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            
            # Initialize Vertex AI client
            aiplatform.init(
                project=self.config["project_id"],
                location=self.config["location"],
                credentials=credentials
            )
            
            self.client = aiplatform
            return self.client
        except Exception as e:
            logger.error(f"Failed to connect to Vertex AI: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Vertex AI: {str(e)}")
        
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of models from Vertex AI."""
        if not self.client:
            self.connect()
            
        try:
            models = self.client.Model.list()
            result = []
            for model in models:
                result.append({
                    "name": model.display_name,
                    "resource_name": model.resource_name,
                    "version_id": model.version_id,
                    "create_time": model.create_time.isoformat() if hasattr(model, "create_time") else None,
                    "update_time": model.update_time.isoformat() if hasattr(model, "update_time") else None
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get models from Vertex AI: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get models from Vertex AI: {str(e)}")
        
    def register_model(self, model_name: str, model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register a model with Vertex AI."""
        if not self.client:
            self.connect()
            
        try:
            # In Vertex AI, we would typically upload a model
            # This is a simplified version that assumes the model is already in GCS
            model = self.client.Model.upload(
                display_name=model_name,
                artifact_uri=metadata.get("artifact_uri", ""),
                serving_container_image_uri=metadata.get("serving_container_image_uri", "")
            )
            
            # Add metadata as labels
            model_labels = {k: str(v) for k, v in metadata.items() 
                           if k not in ["artifact_uri", "serving_container_image_uri"]}
            model.update(labels=model_labels)
            
            return {
                "name": model_name,
                "version": model_version,
                "resource_name": model.resource_name,
                "registered_at": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Failed to register model with Vertex AI: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to register model with Vertex AI: {str(e)}")


def get_connector(integration: Integration) -> BaseConnector:
    """Factory function to get the appropriate connector based on integration type."""
    if integration.integration_type == IntegrationType.MLFLOW:
        return MLflowConnector(integration.config)
    elif integration.integration_type == IntegrationType.SAGEMAKER:
        return SageMakerConnector(integration.config)
    elif integration.integration_type == IntegrationType.VERTEX_AI:
        return VertexAIConnector(integration.config)
    else:
        raise ValueError(f"Unsupported integration type: {integration.integration_type}")


def create_integration(db: Session, integration: IntegrationCreate) -> Integration:
    """Create a new integration."""
    db_integration = Integration(
        name=integration.name,
        integration_type=integration.integration_type,
        config=integration.config,
        is_active=integration.is_active,
        organization_id=integration.organization_id,
        project_id=integration.project_id
    )
    
    # Validate the configuration by attempting to connect
    connector = get_connector(db_integration)
    if not connector.validate_config():
        raise HTTPException(status_code=400, detail="Invalid configuration for integration")
    
    # Test connection
    try:
        connector.connect()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to integration: {str(e)}")
    
    db.add(db_integration)
    db.commit()
    db.refresh(db_integration)
    return db_integration


def update_integration(db: Session, integration_id: int, integration_update: IntegrationUpdate) -> Integration:
    """Update an existing integration."""
    db_integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not db_integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    update_data = integration_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_integration, key, value)
    
    # If config was updated, validate it
    if "config" in update_data:
        connector = get_connector(db_integration)
        if not connector.validate_config():
            raise HTTPException(status_code=400, detail="Invalid configuration for integration")
        
        # Test connection
        try:
            connector.connect()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to connect to integration: {str(e)}")
    
    db.commit()
    db.refresh(db_integration)
    return db_integration


def delete_integration(db: Session, integration_id: int) -> bool:
    """Delete an integration."""
    db_integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not db_integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    db.delete(db_integration)
    db.commit()
    return True


def get_integration(db: Session, integration_id: int) -> Integration:
    """Get an integration by ID."""
    db_integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not db_integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    return db_integration


def get_integrations(db: Session, organization_id: int, project_id: Optional[int] = None, 
                    integration_type: Optional[IntegrationType] = None) -> List[Integration]:
    """Get integrations by organization ID and optionally project ID or type."""
    query = db.query(Integration).filter(Integration.organization_id == organization_id)
    
    if project_id:
        query = query.filter(Integration.project_id == project_id)
    
    if integration_type:
        query = query.filter(Integration.integration_type == integration_type)
    
    return query.all()


def get_models_from_integration(db: Session, integration_id: int) -> List[Dict[str, Any]]:
    """Get models from an integration."""
    db_integration = get_integration(db, integration_id)
    connector = get_connector(db_integration)
    return connector.get_models()


def register_model_with_integration(db: Session, integration_id: int, model_name: str, 
                                   model_version: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Register a model with an integration."""
    db_integration = get_integration(db, integration_id)
    connector = get_connector(db_integration)
    return connector.register_model(model_name, model_version, metadata)