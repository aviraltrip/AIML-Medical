FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn gunicorn

COPY . .

RUN pip install --no-cache-dir .


EXPOSE 7860

CMD ["python", "main.py"]
