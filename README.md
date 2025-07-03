# ModelPulse Backend

ModelPulse is a real-time ML model monitoring platform that allows you to track and analyze model inference data.

## Features

- FastAPI backend with endpoints for logging model inference data
- Database storage for inference logs
- Simulator script to generate synthetic inference data

## Requirements

- Python 3.8+
- PostgreSQL or SQLite (default)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/modelpulse-backend.git
cd modelpulse-backend
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

## Usage

### Running the API

There are multiple ways to start the FastAPI server:

#### Option 1: Using the run.py script

```bash
python run.py
```

#### Option 2: Using uvicorn directly

```bash
uvicorn app.main:app --reload
```

#### Option 3: Using Docker Compose

```bash
docker-compose up
```

The API will be available at http://localhost:8000.

### API Endpoints

- `GET /`: Returns a welcome message
- `POST /log`: Logs model inference data
- `GET /logs`: Retrieves logged data with optional filters

### Running the Simulator

The simulator sends synthetic inference data to the API:

```bash
python simulator.py
```

## Database Configuration

By default, the application uses SQLite. To use PostgreSQL or MySQL, update the database URL in `app/database.py` and `alembic.ini`.

## Project Structure

```
modelpulse-backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   └── database.py      # Database connection
├── migrations/          # Alembic migrations
│   ├── versions/        # Migration versions
│   ├── env.py           # Alembic environment
│   └── script.py.mako   # Migration template
├── alembic.ini          # Alembic configuration
├── simulator.py         # Simulator script
├── run.py               # Entry point script
├── start.sh             # Docker entry point
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## License

MIT
