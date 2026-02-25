# Fast ML-friendly base image
FROM python:3.10-slim

WORKDIR /app

# Install only required system libs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app ./app
COPY src ./src

# Create runtime folders
RUN mkdir -p app/uploads app/results

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]