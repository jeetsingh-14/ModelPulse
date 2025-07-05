# ModelPulse

ModelPulse is a real-time ML model monitoring platform that allows you to track and analyze model inference data.

## Features

### Backend
- FastAPI backend with endpoints for logging model inference data
- Database storage for inference logs
- Analytics endpoints for aggregated metrics
- Alert threshold configuration and detection
- Simulator script to generate synthetic inference data

### Frontend
- React.js dashboard with Ant Design components
- Real-time monitoring of model performance
- Visual analytics with charts and graphs
- Live logs table with filtering and sorting
- Alert system for monitoring model performance
- Settings page for configuring alert thresholds

## Requirements

### Backend
- Python 3.8+
- PostgreSQL or SQLite (default)

### Frontend
- Node.js 14+
- npm or yarn

## Installation

### Backend

1. Clone the repository:

```bash
git clone https://github.com/yourusername/modelpulse.git
cd modelpulse/modelpulse-backend
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Initialize the database:

```bash
alembic upgrade head
```

### Frontend

1. Navigate to the frontend directory:

```bash
cd ../frontend
```

2. Install dependencies:

```bash
npm install
```

3. Create a `.env` file with the backend URL:

```
VITE_API_URL=http://localhost:8000
```

## Usage

### Local Development

#### Running the Backend

There are multiple ways to start the FastAPI server:

##### Option 1: Using the run.py script

```bash
cd modelpulse-backend
python run.py
```

##### Option 2: Using uvicorn directly

```bash
cd modelpulse-backend
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000.

#### Running the Frontend

To start the frontend development server:

```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000.

### Running with Docker Compose

You can run the entire stack (frontend, backend, database, and monitoring) using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- Frontend at http://localhost:80
- Backend API at http://localhost:8000
- PostgreSQL database
- Prometheus at http://localhost:9090
- Grafana at http://localhost:3000
- Node Exporter for host metrics
- cAdvisor for container metrics

### Production Deployment

For production deployment, follow these steps:

1. Update the production environment files:
   - `modelpulse-backend/.env.production`
   - `frontend/.env.production`

2. Build and push the Docker images:
   ```bash
   docker-compose build
   docker tag modelpulse-frontend:latest yourusername/modelpulse-frontend:latest
   docker tag modelpulse-backend:latest yourusername/modelpulse-backend:latest
   docker push yourusername/modelpulse-frontend:latest
   docker push yourusername/modelpulse-backend:latest
   ```

3. Deploy to your chosen cloud platform:
   - **Render**: Set up a Web Service for each component
   - **Heroku**: Use the Heroku Container Registry
   - **AWS ECS**: Deploy using the AWS CLI or console
   - **Azure Web Apps**: Deploy using the Azure CLI or portal

The CI/CD pipeline will automatically build, test, and deploy the application when changes are pushed to the main branch.

### API Endpoints

- `GET /`: Returns a welcome message
- `POST /log`: Logs model inference data
- `GET /logs`: Retrieves logged data with optional filters
- `GET /analytics`: Retrieves analytics data
- `GET /alert-thresholds`: Retrieves alert thresholds
- `POST /alert-thresholds`: Creates a new alert threshold
- `PUT /alert-thresholds/{id}`: Updates an alert threshold
- `DELETE /alert-thresholds/{id}`: Deletes an alert threshold

### Running the Simulator

The simulator sends synthetic inference data to the API:

```bash
cd modelpulse-backend
python simulator.py
```

## Database Configuration

By default, the application uses SQLite. To use PostgreSQL or MySQL, update the database URL in `app/database.py` and `alembic.ini`.

## Monitoring and Logging

ModelPulse includes comprehensive monitoring and logging capabilities:

### Infrastructure Monitoring

- **Prometheus**: Collects metrics from the application and infrastructure
  - Access the Prometheus UI at http://localhost:9090
  - Configured to scrape metrics from the API, Node Exporter, and cAdvisor

- **Grafana**: Visualizes metrics and provides dashboards
  - Access Grafana at http://localhost:3000
  - Default credentials: admin/admin
  - Pre-configured to use Prometheus as a data source

- **Node Exporter**: Collects host-level metrics (CPU, memory, disk, network)

- **cAdvisor**: Collects container-level metrics

### Application Monitoring

- **Sentry**: Tracks and reports application errors
  - Configure by setting the `SENTRY_DSN` in environment variables
  - Integrated with both backend (FastAPI) and frontend (React)

### Setting Up Dashboards

1. Log in to Grafana at http://localhost:3000
2. Go to Dashboards > Import
3. Import dashboards using their IDs:
   - Node Exporter Full: 1860
   - Docker Containers: 893
   - FastAPI Application: 14282

## Project Structure

```
modelpulse/
├── .github/                 # GitHub configuration
│   └── workflows/           # GitHub Actions workflows
│       └── ci-cd.yml        # CI/CD pipeline configuration
├── modelpulse-backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── database.py      # Database connection
│   ├── migrations/          # Alembic migrations
│   │   ├── versions/        # Migration versions
│   │   ├── env.py           # Alembic environment
│   │   └── script.py.mako   # Migration template
│   ├── alembic.ini          # Alembic configuration
│   ├── simulator.py         # Simulator script
│   ├── run.py               # Entry point script
│   ├── start.sh             # Docker entry point
│   ├── Dockerfile           # Docker configuration
│   ├── .env.production      # Production environment variables
│   └── requirements.txt     # Backend dependencies
├── frontend/
│   ├── public/              # Static assets
│   ├── src/
│   │   ├── components/      # Reusable components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   ├── types/           # TypeScript interfaces
│   │   ├── App.tsx          # Main application component
│   │   ├── main.tsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── .env                 # Development environment variables
│   ├── .env.production      # Production environment variables
│   ├── Dockerfile           # Docker configuration
│   ├── package.json         # Frontend dependencies
│   ├── tsconfig.json        # TypeScript configuration
│   ├── vite.config.ts       # Vite configuration
│   └── README.md            # Frontend documentation
├── docker-compose.yml       # Docker Compose for full stack
├── prometheus.yml           # Prometheus configuration
└── README.md                # This file
```

## License

MIT
