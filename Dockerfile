FROM python:3.10-slim

# Set working directory
WORKDIR /app


# Installing system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements first (for caching)
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.helmet.app.app:app", "--host", "0.0.0.0", "--port", "8000"]