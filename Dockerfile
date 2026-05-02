# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies (Tesseract OCR, etc.)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir uvicorn gunicorn

# Copy project files
COPY . .

# Expose the port (HF Spaces uses 7860)
EXPOSE 7860

# Command to run the application
CMD ["python", "main.py"]
