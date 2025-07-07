# ModelPulse Integrations Guide

This document provides information about the integrations, webhooks, and export capabilities added to ModelPulse in Phase 7.

## Table of Contents

- [ML Platform Integrations](#ml-platform-integrations)
- [Log Export Capabilities](#log-export-capabilities)
- [Webhooks](#webhooks)
- [API Reference](#api-reference)

## ML Platform Integrations

ModelPulse can now integrate with popular ML model registries and deployment platforms:

### MLflow

Connect ModelPulse to your MLflow Tracking Server to:
- Pull model metadata
- Register monitored models
- Track model versions

**Configuration Requirements:**
- Tracking URI (e.g., `http://mlflow-server:5000`)

### Amazon SageMaker

Connect ModelPulse to AWS SageMaker to:
- List deployed models
- Register new models
- Monitor SageMaker endpoints

**Configuration Requirements:**
- AWS Access Key ID
- AWS Secret Access Key
- Region Name

### Google Vertex AI

Connect ModelPulse to Google Vertex AI to:
- List deployed models
- Register new models
- Monitor Vertex AI model endpoints

**Configuration Requirements:**
- Project ID
- Location (e.g., `us-central1`)
- Service Account Credentials (JSON)

## Log Export Capabilities

ModelPulse can export logs and metrics to enterprise monitoring tools:

### ELK Stack (Elasticsearch, Logstash, Kibana)

Export inference logs and drift metrics to Elasticsearch for visualization in Kibana.

**Configuration Requirements:**
- Elasticsearch URL
- Username
- Password
- Optional: Verify SSL certificates (boolean)
- Optional: Custom index names

### Splunk

Export data to Splunk via HTTP Event Collector (HEC).

**Configuration Requirements:**
- HEC URL
- HEC Token
- Optional: Verify SSL certificates (boolean)

### Datadog

Export events and metrics to Datadog.

**Configuration Requirements:**
- API Key
- Application Key

## Webhooks

ModelPulse can trigger external workflows via webhooks on specific events:

### Supported Events

- **Drift Detected**: Triggered when model drift is detected
- **Threshold Breached**: Triggered when a model metric exceeds a defined threshold
- **Recommendation Created**: Triggered when a new recommendation is generated
- **Model Registered**: Triggered when a new model is registered with an integration

### Webhook Configuration

Webhooks can be configured with:
- Name
- URL
- Events to listen for
- Custom headers (optional)
- Organization and project scope

### Webhook Payload

Webhook payloads include:
- Event type
- Timestamp
- Event-specific data

## API Reference

### Integrations API

- `GET /organizations/{organization_id}/integrations` - List all integrations
- `GET /organizations/{organization_id}/integrations/{integration_id}` - Get integration details
- `POST /organizations/{organization_id}/integrations` - Create a new integration
- `PUT /organizations/{organization_id}/integrations/{integration_id}` - Update an integration
- `DELETE /organizations/{organization_id}/integrations/{integration_id}` - Delete an integration
- `GET /organizations/{organization_id}/integrations/{integration_id}/models` - List models from integration
- `POST /organizations/{organization_id}/integrations/{integration_id}/register-model` - Register a model

### Export API

- `GET /organizations/{organization_id}/exports` - List all export configurations
- `GET /organizations/{organization_id}/exports/{export_id}` - Get export configuration details
- `POST /organizations/{organization_id}/exports` - Create a new export configuration
- `PUT /organizations/{organization_id}/exports/{export_id}` - Update an export configuration
- `DELETE /organizations/{organization_id}/exports/{export_id}` - Delete an export configuration
- `POST /organizations/{organization_id}/exports/{export_id}/trigger` - Trigger an export

### Webhooks API

- `GET /organizations/{organization_id}/webhooks` - List all webhooks
- `GET /organizations/{organization_id}/webhooks/{webhook_id}` - Get webhook details
- `POST /organizations/{organization_id}/webhooks` - Create a new webhook
- `PUT /organizations/{organization_id}/webhooks/{webhook_id}` - Update a webhook
- `DELETE /organizations/{organization_id}/webhooks/{webhook_id}` - Delete a webhook
- `GET /organizations/{organization_id}/webhooks/{webhook_id}/history` - Get webhook invocation history