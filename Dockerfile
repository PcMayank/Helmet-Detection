# Use smaller Python image
FROM python:3.10-slim

WORKDIR /app

# Install only required system libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install deps (no cache)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make folders
RUN mkdir -p uploads results

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]