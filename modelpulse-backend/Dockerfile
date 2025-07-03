FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose the port
EXPOSE 8000

# Start the application using the start script
CMD ["./start.sh"]
