FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# COPY api/ ./api/
COPY src/api/ ./api/

# Expose FastAPI's default port
EXPOSE 8000

# FIX: Point to the actual module paths correctly.
# 'api.main:app' translates to: look in the 'api' directory, open 'main.py', find 'app = FastAPI()'
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]