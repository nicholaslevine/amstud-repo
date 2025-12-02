FROM python:3.11-slim

# Install system dependencies needed by PyMuPDF, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Railway sets $PORT automatically; make sure your app uses it
ENV PORT=8000

# Start the app
# If your file is named something else, change "backend:app" accordingly
CMD ["python", "backend.py"]
