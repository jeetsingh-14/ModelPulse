#!/bin/sh

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Start the API server
echo "Starting API server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload