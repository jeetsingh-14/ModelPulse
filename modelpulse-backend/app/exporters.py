"""
Exporters module for sending ModelPulse logs to external monitoring systems.

This module provides exporters for:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk HEC (HTTP Event Collector)
- Datadog API
"""

from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session
from fastapi import HTTPException
import json
import logging
import requests
from datetime import datetime, timedelta
import time

from .models import ExportConfiguration, ExportType, InferenceLog, DriftMetrics
from .schemas import ExportConfigurationCreate, ExportConfigurationUpdate

# Set up logging
logger = logging.getLogger(__name__)


class BaseExporter:
    """Base class for all log exporters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def validate_config(self) -> bool:
        """Validate the configuration."""
        raise NotImplementedError("Subclasses must implement validate_config")
        
    def test_connection(self) -> bool:
        """Test connection to the export destination."""
        raise NotImplementedError("Subclasses must implement test_connection")
        
    def export_inference_logs(self, logs: List[InferenceLog]) -> bool:
        """Export inference logs to the destination."""
        raise NotImplementedError("Subclasses must implement export_inference_logs")
        
    def export_drift_metrics(self, metrics: List[DriftMetrics]) -> bool:
        """Export drift metrics to the destination."""
        raise NotImplementedError("Subclasses must implement export_drift_metrics")


class ELKExporter(BaseExporter):
    """Exporter for ELK Stack (Elasticsearch, Logstash, Kibana)."""
    
    def validate_config(self) -> bool:
        """Validate ELK configuration."""
        required_keys = ["elasticsearch_url", "username", "password"]
        return all(key in self.config for key in required_keys)
        
    def test_connection(self) -> bool:
        """Test connection to Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch
            
            es = Elasticsearch(
                [self.config["elasticsearch_url"]],
                http_auth=(self.config["username"], self.config["password"]),
                verify_certs=self.config.get("verify_certs", True)
            )
            
            # Check if the cluster is up
            health = es.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            return False
        
    def export_inference_logs(self, logs: List[InferenceLog]) -> bool:
        """Export inference logs to Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch.helpers import bulk
            
            es = Elasticsearch(
                [self.config["elasticsearch_url"]],
                http_auth=(self.config["username"], self.config["password"]),
                verify_certs=self.config.get("verify_certs", True)
            )
            
            # Prepare index name
            index_name = self.config.get("inference_index", "modelpulse-inference-logs")
            
            # Prepare documents for bulk indexing
            actions = []
            for log in logs:
                doc = {
                    "_index": index_name,
                    "_id": log.id,
                    "_source": {
                        "model_name": log.model_name,
                        "timestamp": log.timestamp.isoformat(),
                        "input_shape": log.input_shape,
                        "latency_ms": log.latency_ms,
                        "confidence": log.confidence,
                        "output_class": log.output_class,
                        "organization_id": log.organization_id,
                        "project_id": log.project_id
                    }
                }
                actions.append(doc)
            
            # Bulk index the documents
            success, failed = bulk(es, actions, refresh=True)
            
            logger.info(f"Exported {success} inference logs to Elasticsearch, {len(failed)} failed")
            return len(failed) == 0
        except Exception as e:
            logger.error(f"Failed to export inference logs to Elasticsearch: {str(e)}")
            return False
        
    def export_drift_metrics(self, metrics: List[DriftMetrics]) -> bool:
        """Export drift metrics to Elasticsearch."""
        try:
            from elasticsearch import Elasticsearch
            from elasticsearch.helpers import bulk
            
            es = Elasticsearch(
                [self.config["elasticsearch_url"]],
                http_auth=(self.config["username"], self.config["password"]),
                verify_certs=self.config.get("verify_certs", True)
            )
            
            # Prepare index name
            index_name = self.config.get("drift_index", "modelpulse-drift-metrics")
            
            # Prepare documents for bulk indexing
            actions = []
            for metric in metrics:
                doc = {
                    "_index": index_name,
                    "_id": metric.id,
                    "_source": {
                        "model_name": metric.model_name,
                        "timestamp": metric.timestamp.isoformat(),
                        "input_kl_divergence": metric.input_kl_divergence,
                        "input_psi": metric.input_psi,
                        "output_kl_divergence": metric.output_kl_divergence,
                        "output_psi": metric.output_psi,
                        "drift_severity": metric.drift_severity,
                        "explanation": metric.explanation,
                        "organization_id": metric.organization_id,
                        "project_id": metric.project_id
                    }
                }
                actions.append(doc)
            
            # Bulk index the documents
            success, failed = bulk(es, actions, refresh=True)
            
            logger.info(f"Exported {success} drift metrics to Elasticsearch, {len(failed)} failed")
            return len(failed) == 0
        except Exception as e:
            logger.error(f"Failed to export drift metrics to Elasticsearch: {str(e)}")
            return False


class SplunkExporter(BaseExporter):
    """Exporter for Splunk HEC (HTTP Event Collector)."""
    
    def validate_config(self) -> bool:
        """Validate Splunk configuration."""
        required_keys = ["hec_url", "hec_token"]
        return all(key in self.config for key in required_keys)
        
    def test_connection(self) -> bool:
        """Test connection to Splunk HEC."""
        try:
            url = self.config["hec_url"]
            headers = {
                "Authorization": f"Splunk {self.config['hec_token']}",
                "Content-Type": "application/json"
            }
            
            # Send a test event
            data = {
                "event": "ModelPulse connection test",
                "source": "modelpulse",
                "sourcetype": "modelpulse:test"
            }
            
            response = requests.post(
                url, 
                headers=headers, 
                json=data,
                verify=self.config.get("verify_ssl", True)
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Splunk HEC: {str(e)}")
            return False
        
    def export_inference_logs(self, logs: List[InferenceLog]) -> bool:
        """Export inference logs to Splunk."""
        try:
            url = self.config["hec_url"]
            headers = {
                "Authorization": f"Splunk {self.config['hec_token']}",
                "Content-Type": "application/json"
            }
            
            # Prepare events for Splunk
            events = []
            for log in logs:
                event = {
                    "time": int(log.timestamp.timestamp()),
                    "host": "modelpulse",
                    "source": "modelpulse",
                    "sourcetype": "modelpulse:inference",
                    "event": {
                        "id": log.id,
                        "model_name": log.model_name,
                        "input_shape": log.input_shape,
                        "latency_ms": log.latency_ms,
                        "confidence": log.confidence,
                        "output_class": log.output_class,
                        "organization_id": log.organization_id,
                        "project_id": log.project_id
                    }
                }
                events.append(event)
            
            # Send events to Splunk
            success_count = 0
            for event in events:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=event,
                    verify=self.config.get("verify_ssl", True)
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    logger.error(f"Failed to export event to Splunk: {response.text}")
            
            logger.info(f"Exported {success_count}/{len(events)} inference logs to Splunk")
            return success_count == len(events)
        except Exception as e:
            logger.error(f"Failed to export inference logs to Splunk: {str(e)}")
            return False
        
    def export_drift_metrics(self, metrics: List[DriftMetrics]) -> bool:
        """Export drift metrics to Splunk."""
        try:
            url = self.config["hec_url"]
            headers = {
                "Authorization": f"Splunk {self.config['hec_token']}",
                "Content-Type": "application/json"
            }
            
            # Prepare events for Splunk
            events = []
            for metric in metrics:
                event = {
                    "time": int(metric.timestamp.timestamp()),
                    "host": "modelpulse",
                    "source": "modelpulse",
                    "sourcetype": "modelpulse:drift",
                    "event": {
                        "id": metric.id,
                        "model_name": metric.model_name,
                        "input_kl_divergence": metric.input_kl_divergence,
                        "input_psi": metric.input_psi,
                        "output_kl_divergence": metric.output_kl_divergence,
                        "output_psi": metric.output_psi,
                        "drift_severity": metric.drift_severity,
                        "explanation": metric.explanation,
                        "organization_id": metric.organization_id,
                        "project_id": metric.project_id
                    }
                }
                events.append(event)
            
            # Send events to Splunk
            success_count = 0
            for event in events:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=event,
                    verify=self.config.get("verify_ssl", True)
                )
                
                if response.status_code == 200:
                    success_count += 1
                else:
                    logger.error(f"Failed to export event to Splunk: {response.text}")
            
            logger.info(f"Exported {success_count}/{len(events)} drift metrics to Splunk")
            return success_count == len(events)
        except Exception as e:
            logger.error(f"Failed to export drift metrics to Splunk: {str(e)}")
            return False


class DatadogExporter(BaseExporter):
    """Exporter for Datadog API."""
    
    def validate_config(self) -> bool:
        """Validate Datadog configuration."""
        required_keys = ["api_key", "app_key"]
        return all(key in self.config for key in required_keys)
        
    def test_connection(self) -> bool:
        """Test connection to Datadog API."""
        try:
            from datadog import initialize, api
            
            # Initialize the Datadog API
            options = {
                'api_key': self.config["api_key"],
                'app_key': self.config["app_key"]
            }
            initialize(**options)
            
            # Test connection by getting API info
            api_info = api.ApiClient.api_client.get_api_client().get_info()
            return api_info is not None
        except Exception as e:
            logger.error(f"Failed to connect to Datadog API: {str(e)}")
            return False
        
    def export_inference_logs(self, logs: List[InferenceLog]) -> bool:
        """Export inference logs to Datadog."""
        try:
            from datadog import initialize, api
            
            # Initialize the Datadog API
            options = {
                'api_key': self.config["api_key"],
                'app_key': self.config["app_key"]
            }
            initialize(**options)
            
            # Prepare events for Datadog
            success_count = 0
            for log in logs:
                # Create an event
                title = f"ModelPulse Inference: {log.model_name}"
                text = f"""
                Model: {log.model_name}
                Output Class: {log.output_class}
                Confidence: {log.confidence}
                Latency: {log.latency_ms} ms
                Organization ID: {log.organization_id}
                Project ID: {log.project_id or 'N/A'}
                """
                
                tags = [
                    f"model:{log.model_name}",
                    f"output_class:{log.output_class}",
                    f"organization_id:{log.organization_id}"
                ]
                
                if log.project_id:
                    tags.append(f"project_id:{log.project_id}")
                
                # Post the event
                event = api.Event.create(
                    title=title,
                    text=text,
                    tags=tags,
                    date_happened=int(log.timestamp.timestamp())
                )
                
                if event and 'event' in event:
                    success_count += 1
                    
                # Also send metrics
                api.Metric.send(
                    metric="modelpulse.inference.latency",
                    points=[(int(log.timestamp.timestamp()), log.latency_ms)],
                    tags=tags
                )
                
                api.Metric.send(
                    metric="modelpulse.inference.confidence",
                    points=[(int(log.timestamp.timestamp()), log.confidence)],
                    tags=tags
                )
            
            logger.info(f"Exported {success_count}/{len(logs)} inference logs to Datadog")
            return success_count == len(logs)
        except Exception as e:
            logger.error(f"Failed to export inference logs to Datadog: {str(e)}")
            return False
        
    def export_drift_metrics(self, metrics: List[DriftMetrics]) -> bool:
        """Export drift metrics to Datadog."""
        try:
            from datadog import initialize, api
            
            # Initialize the Datadog API
            options = {
                'api_key': self.config["api_key"],
                'app_key': self.config["app_key"]
            }
            initialize(**options)
            
            # Prepare events for Datadog
            success_count = 0
            for metric in metrics:
                # Create an event
                title = f"ModelPulse Drift: {metric.model_name} - {metric.drift_severity}"
                text = f"""
                Model: {metric.model_name}
                Severity: {metric.drift_severity}
                Input KL Divergence: {metric.input_kl_divergence}
                Input PSI: {metric.input_psi}
                Output KL Divergence: {metric.output_kl_divergence}
                Output PSI: {metric.output_psi}
                Explanation: {metric.explanation or 'N/A'}
                Organization ID: {metric.organization_id}
                Project ID: {metric.project_id or 'N/A'}
                """
                
                tags = [
                    f"model:{metric.model_name}",
                    f"severity:{metric.drift_severity}",
                    f"organization_id:{metric.organization_id}"
                ]
                
                if metric.project_id:
                    tags.append(f"project_id:{metric.project_id}")
                
                # Post the event
                event = api.Event.create(
                    title=title,
                    text=text,
                    tags=tags,
                    date_happened=int(metric.timestamp.timestamp()),
                    alert_type="warning" if metric.drift_severity != "OK" else "info"
                )
                
                if event and 'event' in event:
                    success_count += 1
                    
                # Also send metrics
                api.Metric.send(
                    metric="modelpulse.drift.input_kl_divergence",
                    points=[(int(metric.timestamp.timestamp()), metric.input_kl_divergence)],
                    tags=tags
                )
                
                api.Metric.send(
                    metric="modelpulse.drift.input_psi",
                    points=[(int(metric.timestamp.timestamp()), metric.input_psi)],
                    tags=tags
                )
                
                api.Metric.send(
                    metric="modelpulse.drift.output_kl_divergence",
                    points=[(int(metric.timestamp.timestamp()), metric.output_kl_divergence)],
                    tags=tags
                )
                
                api.Metric.send(
                    metric="modelpulse.drift.output_psi",
                    points=[(int(metric.timestamp.timestamp()), metric.output_psi)],
                    tags=tags
                )
            
            logger.info(f"Exported {success_count}/{len(metrics)} drift metrics to Datadog")
            return success_count == len(metrics)
        except Exception as e:
            logger.error(f"Failed to export drift metrics to Datadog: {str(e)}")
            return False


def get_exporter(export_config: ExportConfiguration) -> BaseExporter:
    """Factory function to get the appropriate exporter based on export type."""
    if export_config.export_type == ExportType.ELK:
        return ELKExporter(export_config.config)
    elif export_config.export_type == ExportType.SPLUNK:
        return SplunkExporter(export_config.config)
    elif export_config.export_type == ExportType.DATADOG:
        return DatadogExporter(export_config.config)
    else:
        raise ValueError(f"Unsupported export type: {export_config.export_type}")


def create_export_config(db: Session, export_config: ExportConfigurationCreate) -> ExportConfiguration:
    """Create a new export configuration."""
    db_export_config = ExportConfiguration(
        name=export_config.name,
        export_type=export_config.export_type,
        config=export_config.config,
        schedule=export_config.schedule,
        is_active=export_config.is_active,
        organization_id=export_config.organization_id,
        project_id=export_config.project_id
    )
    
    # Validate the configuration by attempting to connect
    exporter = get_exporter(db_export_config)
    if not exporter.validate_config():
        raise HTTPException(status_code=400, detail="Invalid configuration for export")
    
    # Test connection
    if not exporter.test_connection():
        raise HTTPException(status_code=400, detail="Failed to connect to export destination")
    
    db.add(db_export_config)
    db.commit()
    db.refresh(db_export_config)
    return db_export_config


def update_export_config(db: Session, export_config_id: int, export_config_update: ExportConfigurationUpdate) -> ExportConfiguration:
    """Update an existing export configuration."""
    db_export_config = db.query(ExportConfiguration).filter(ExportConfiguration.id == export_config_id).first()
    if not db_export_config:
        raise HTTPException(status_code=404, detail="Export configuration not found")
    
    update_data = export_config_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_export_config, key, value)
    
    # If config was updated, validate it
    if "config" in update_data:
        exporter = get_exporter(db_export_config)
        if not exporter.validate_config():
            raise HTTPException(status_code=400, detail="Invalid configuration for export")
        
        # Test connection
        if not exporter.test_connection():
            raise HTTPException(status_code=400, detail="Failed to connect to export destination")
    
    db.commit()
    db.refresh(db_export_config)
    return db_export_config


def delete_export_config(db: Session, export_config_id: int) -> bool:
    """Delete an export configuration."""
    db_export_config = db.query(ExportConfiguration).filter(ExportConfiguration.id == export_config_id).first()
    if not db_export_config:
        raise HTTPException(status_code=404, detail="Export configuration not found")
    
    db.delete(db_export_config)
    db.commit()
    return True


def get_export_config(db: Session, export_config_id: int) -> ExportConfiguration:
    """Get an export configuration by ID."""
    db_export_config = db.query(ExportConfiguration).filter(ExportConfiguration.id == export_config_id).first()
    if not db_export_config:
        raise HTTPException(status_code=404, detail="Export configuration not found")
    return db_export_config


def get_export_configs(db: Session, organization_id: int, project_id: Optional[int] = None, 
                      export_type: Optional[ExportType] = None) -> List[ExportConfiguration]:
    """Get export configurations by organization ID and optionally project ID or type."""
    query = db.query(ExportConfiguration).filter(ExportConfiguration.organization_id == organization_id)
    
    if project_id:
        query = query.filter(ExportConfiguration.project_id == project_id)
    
    if export_type:
        query = query.filter(ExportConfiguration.export_type == export_type)
    
    return query.all()


def export_logs(db: Session, export_config_id: int, start_time: Optional[datetime] = None, 
               end_time: Optional[datetime] = None) -> bool:
    """Export logs using a specific export configuration."""
    db_export_config = get_export_config(db, export_config_id)
    exporter = get_exporter(db_export_config)
    
    # Set default time range if not provided
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(hours=1)
    
    # Get logs to export
    query = db.query(InferenceLog).filter(
        InferenceLog.organization_id == db_export_config.organization_id,
        InferenceLog.timestamp >= start_time,
        InferenceLog.timestamp <= end_time
    )
    
    if db_export_config.project_id:
        query = query.filter(InferenceLog.project_id == db_export_config.project_id)
    
    logs = query.all()
    
    # Get drift metrics to export
    query = db.query(DriftMetrics).filter(
        DriftMetrics.organization_id == db_export_config.organization_id,
        DriftMetrics.timestamp >= start_time,
        DriftMetrics.timestamp <= end_time
    )
    
    if db_export_config.project_id:
        query = query.filter(DriftMetrics.project_id == db_export_config.project_id)
    
    metrics = query.all()
    
    # Export logs and metrics
    logs_success = exporter.export_inference_logs(logs) if logs else True
    metrics_success = exporter.export_drift_metrics(metrics) if metrics else True
    
    return logs_success and metrics_success