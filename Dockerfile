FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the application code
# COPY api/ ./api/
COPY src/api/ ./api/

# Expose FastAPI's default port
EXPOSE 8000

# FIX: Point to the actual module paths correctly.
# 'api.main:app' translates to: look in the 'api' directory, open 'main.py', find 'app = FastAPI()'
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]