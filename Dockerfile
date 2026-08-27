FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY src/inference.py ./src/inference.py

# Copy trained model
COPY models/baseline_cnn.pt ./models/baseline_cnn.pt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]