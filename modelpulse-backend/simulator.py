import requests
import random
import time
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API endpoint - use environment variable or default
API_URL = os.getenv("API_URL", "http://localhost:8000/log")
logger.info(f"Using API URL: {API_URL}")

# Model names
MODEL_NAMES = ["classifier-v1", "detector-v2", "segmenter-v1"]

# Output classes
OUTPUT_CLASSES = ["cat", "dog", "bird", "car", "person", "bicycle"]

# Input shapes
INPUT_SHAPES = [
    [1, 224, 224, 3],  # Standard image
    [1, 299, 299, 3],  # Inception-style
    [1, 512, 512, 3],  # High-res
    [1, 128, 128, 3],  # Low-res
]

def generate_random_inference_data():
    """Generate random inference data."""
    return {
        "model_name": random.choice(MODEL_NAMES),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input_shape": random.choice(INPUT_SHAPES),
        "latency_ms": round(random.uniform(10, 500), 2),
        "confidence": round(random.uniform(0.1, 0.99), 2),
        "output_class": random.choice(OUTPUT_CLASSES)
    }

def send_inference_data():
    """Send inference data to the API."""
    data = generate_random_inference_data()

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            logger.info(f"Successfully sent data: {json.dumps(data)}")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"Failed to send data. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error sending data: {str(e)}")

def main():
    """Main function to run the simulator."""
    logger.info("Starting ModelPulse simulator...")

    while True:
        send_inference_data()
        # Sleep for a random time between 5 and 10 seconds
        sleep_time = random.uniform(5, 10)
        logger.info(f"Sleeping for {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
