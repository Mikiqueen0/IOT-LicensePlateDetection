# Use a smaller base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first for dependency installation
COPY requirements.txt .

# Install dependencies with optimizations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 libglib2.0-dev \
    libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

COPY best.pt /app/best.pt

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
