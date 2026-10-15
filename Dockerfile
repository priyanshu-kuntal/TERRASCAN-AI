# Production Dockerfile for TERRASCAN-AI
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose standard port
EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Start Flask Mission Server
CMD ["python", "app.py"]
