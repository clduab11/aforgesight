# Enterprise Retail Analytics Suite
# Production-ready Docker image

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/outputs/plots /app/outputs/reports

# Note: Sample data generation removed from production image
# To generate sample data, run: python data/generate_sample_data.py

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "run_analytics.py", "--task", "all", "--output", "/app/outputs"]
