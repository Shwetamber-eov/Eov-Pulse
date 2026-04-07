# Use a small, efficient Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (curl is useful for health checks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY app/ .

# Expose the port FastAPI will run on
EXPOSE 8000
EXPOSE 8501

# Start the Researcher Agent
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "main.py"]
