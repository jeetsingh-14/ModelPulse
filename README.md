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

### Running the Backend

There are multiple ways to start the FastAPI server:

#### Option 1: Using the run.py script

```bash
cd modelpulse-backend
python run.py
```

#### Option 2: Using uvicorn directly

```bash
cd modelpulse-backend
uvicorn app.main:app --reload
```

#### Option 3: Using Docker Compose

```bash
cd modelpulse-backend
docker-compose up
```

The API will be available at http://localhost:8000.

### Running the Frontend

To start the frontend development server:

```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:3000.

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

## Project Structure

```
modelpulse/
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
│   ├── docker-compose.yml   # Docker Compose configuration
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
│   ├── .env                 # Environment variables
│   ├── package.json         # Frontend dependencies
│   ├── tsconfig.json        # TypeScript configuration
│   ├── vite.config.ts       # Vite configuration
│   └── README.md            # Frontend documentation
└── README.md                # This file
```

## License

MIT
