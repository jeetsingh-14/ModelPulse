# ModelPulse Deployment Guide

This document provides detailed instructions for deploying the ModelPulse application to various cloud environments, setting up CI/CD pipelines, and configuring monitoring.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment Options](#cloud-deployment-options)
  - [Render](#render)
  - [Heroku](#heroku)
  - [AWS ECS](#aws-ecs)
  - [Azure Web Apps](#azure-web-apps)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring Setup](#monitoring-setup)
  - [Prometheus](#prometheus)
  - [Grafana](#grafana)
  - [Sentry](#sentry)
- [Environment Variables](#environment-variables)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying ModelPulse, ensure you have:

- Docker and Docker Compose installed
- Access to your chosen cloud platform
- Docker Hub account (or other container registry)
- GitHub account for CI/CD
- Sentry account for error tracking (optional)

## Docker Deployment

The simplest way to deploy ModelPulse is using Docker Compose:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/modelpulse.git
   cd modelpulse
   ```

2. Configure environment variables:
   - Create `.env.production` files in both `frontend/` and `modelpulse-backend/` directories
   - Set appropriate values for production (see [Environment Variables](#environment-variables))

3. Build and run the containers:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://your-server-ip
   - API: http://your-server-ip:8000
   - Grafana: http://your-server-ip:3000
   - Prometheus: http://your-server-ip:9090

## Cloud Deployment Options

### Render

Render offers a straightforward deployment process:

1. Create a new Web Service for the backend:
   - Connect your GitHub repository
   - Select the `modelpulse-backend` directory
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables

2. Create a new Web Service for the frontend:
   - Connect your GitHub repository
   - Select the `frontend` directory
   - Set the build command: `npm ci && npm run build`
   - Set the start command: `npx serve -s dist -l $PORT`
   - Add environment variables including `VITE_API_URL` pointing to your backend URL

3. Create a PostgreSQL database:
   - Use Render's managed PostgreSQL service
   - Connect it to your backend service

4. Set up the deploy hook:
   - Go to your backend service dashboard
   - Copy the deploy hook URL
   - Add it as a secret in your GitHub repository (`RENDER_DEPLOY_HOOK`)

### Heroku

To deploy to Heroku:

1. Install the Heroku CLI and log in:
   ```bash
   heroku login
   ```

2. Create applications for frontend and backend:
   ```bash
   heroku create modelpulse-backend
   heroku create modelpulse-frontend
   ```

3. Add a PostgreSQL database:
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev --app modelpulse-backend
   ```

4. Configure environment variables:
   ```bash
   heroku config:set VARIABLE_NAME=value --app modelpulse-backend
   heroku config:set VITE_API_URL=https://modelpulse-backend.herokuapp.com --app modelpulse-frontend
   ```

5. Deploy using Docker:
   ```bash
   heroku container:push web --app modelpulse-backend
   heroku container:release web --app modelpulse-backend
   heroku container:push web --app modelpulse-frontend
   heroku container:release web --app modelpulse-frontend
   ```

### AWS ECS

For AWS ECS deployment:

1. Create an ECR repository for each container:
   ```bash
   aws ecr create-repository --repository-name modelpulse-backend
   aws ecr create-repository --repository-name modelpulse-frontend
   ```

2. Build, tag, and push the images:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
   docker build -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/modelpulse-backend:latest ./modelpulse-backend
   docker build -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/modelpulse-frontend:latest ./frontend
   docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/modelpulse-backend:latest
   docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/modelpulse-frontend:latest
   ```

3. Create an ECS cluster, task definitions, and services using the AWS console or CLI
4. Set up an Application Load Balancer to route traffic
5. Configure environment variables in the task definitions

### Azure Web Apps

To deploy to Azure Web Apps:

1. Create Azure Container Registry:
   ```bash
   az acr create --resource-group myResourceGroup --name myContainerRegistry --sku Basic
   ```

2. Build and push images:
   ```bash
   az acr login --name myContainerRegistry
   docker build -t mycontainerregistry.azurecr.io/modelpulse-backend:latest ./modelpulse-backend
   docker build -t mycontainerregistry.azurecr.io/modelpulse-frontend:latest ./frontend
   docker push mycontainerregistry.azurecr.io/modelpulse-backend:latest
   docker push mycontainerregistry.azurecr.io/modelpulse-frontend:latest
   ```

3. Create Web Apps:
   ```bash
   az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name modelpulse-backend --deployment-container-image-name mycontainerregistry.azurecr.io/modelpulse-backend:latest
   az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name modelpulse-frontend --deployment-container-image-name mycontainerregistry.azurecr.io/modelpulse-frontend:latest
   ```

4. Configure environment variables:
   ```bash
   az webapp config appsettings set --resource-group myResourceGroup --name modelpulse-backend --settings VARIABLE_NAME=value
   az webapp config appsettings set --resource-group myResourceGroup --name modelpulse-frontend --settings VITE_API_URL=https://modelpulse-backend.azurewebsites.net
   ```

## CI/CD Pipeline

ModelPulse uses GitHub Actions for CI/CD. The workflow is defined in `.github/workflows/ci-cd.yml`:

1. **Testing**: Runs tests for both frontend and backend
2. **Building**: Builds Docker images for both components
3. **Deployment**: Deploys to the configured cloud platform

To set up the CI/CD pipeline:

1. Add the following secrets to your GitHub repository:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token
   - `RENDER_DEPLOY_HOOK`: The Render deploy hook URL (if using Render)
   - Cloud platform-specific secrets (if using AWS, Azure, etc.)

2. Push changes to the main branch to trigger the pipeline

## Monitoring Setup

### Prometheus

Prometheus is configured to scrape metrics from:
- The FastAPI backend
- Node Exporter (system metrics)
- cAdvisor (container metrics)

The configuration is defined in `prometheus.yml`.

### Grafana

Grafana is pre-configured to use Prometheus as a data source. To set up dashboards:

1. Access Grafana at http://your-server-ip:3000 (default credentials: admin/admin)
2. Go to Dashboards > Import
3. Import the following dashboards by ID:
   - Node Exporter Full: 1860
   - Docker Containers: 893
   - FastAPI Application: 14282

### Sentry

To configure Sentry for error tracking:

1. Create a Sentry account and project
2. Add your Sentry DSN to the environment variables:
   - Backend: `SENTRY_DSN` in `modelpulse-backend/.env.production`
   - Frontend: `VITE_SENTRY_DSN` in `frontend/.env.production`

## Environment Variables

### Backend

Create `modelpulse-backend/.env.production` with:

```
DATABASE_URL=postgresql://username:password@host:port/database
SECRET_KEY=your-secret-key
SENTRY_DSN=your-sentry-dsn
CORS_ORIGINS=https://your-frontend-url.com
```

### Frontend

Create `frontend/.env.production` with:

```
VITE_API_URL=https://your-backend-url.com
VITE_SENTRY_DSN=your-sentry-dsn
```

## Security Considerations

1. **Environment Variables**: Never commit `.env.production` files to the repository
2. **Database Security**: Use strong passwords and restrict network access
3. **API Security**: Implement rate limiting and proper authentication
4. **CORS Configuration**: Restrict allowed origins to your frontend domain
5. **Secrets Management**: Use your cloud provider's secrets management service

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check the `DATABASE_URL` environment variable
   - Ensure the database is accessible from the backend container
   - Verify that migrations have run successfully

2. **CORS Errors**:
   - Check that `CORS_ORIGINS` includes your frontend URL
   - Ensure the frontend is using the correct backend URL

3. **Container Startup Failures**:
   - Check container logs: `docker-compose logs -f`
   - Verify that all required environment variables are set

4. **CI/CD Pipeline Failures**:
   - Check GitHub Actions logs for detailed error messages
   - Verify that all required secrets are configured correctly