# Use Python 3.9 slim base image
FROM python:3.9-slim-bullseye

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Update system packages and clean up
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install -e . \
    && pip cache purge

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser