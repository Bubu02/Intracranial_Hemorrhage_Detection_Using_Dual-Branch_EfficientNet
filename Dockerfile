# Base image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements-docker.txt .

# Install dependencies (CPU-only PyTorch to save space)
RUN pip install --no-cache-dir -r requirements-docker.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn (Production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
