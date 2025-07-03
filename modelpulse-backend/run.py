"""
Entry point for the ModelPulse application.
This script makes it easy to run the application without using Docker.
"""
import uvicorn

if __name__ == "__main__":
    print("Starting ModelPulse API...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)