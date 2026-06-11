# Dockerfile for SmartHouse AI Platform

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling C-extensions (XGBoost, LightGBM, psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Copy all source files
COPY . /app

EXPOSE 8000

# Run uvicorn web server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
