# FastAPI application
import os
import logging
from contextlib import asynccontextmanager
import mlflow
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles API startup and shutdown events. 
    Simplifies and de-duplicates model-loading lifecycle code.
    """
    global model
    model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/CreditRiskModel/Production")
    logger.info(f"Attempting to load model from: {model_uri}")
    
    try:
        # Explicit error handling for model loading
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully into memory.")
    except Exception as e:
        logger.critical(f"Failed to load model on startup: {str(e)}")
        # Note: Depending on your CI/CD, you might want to raise the error 
        # to prevent an unhealthy container from deploying.
        model = None
        
    yield
    # Clean up on shutdown if necessary
    logger.info("Shutting down API...")

app = FastAPI(
    title="Credit Risk Scoring API", 
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Request/Response Models ---
# (Keep this clean or import from your corrected pydantic_models.py)
class CreditApplication(BaseModel):
    income: float = Field(..., gt=0, description="Annual income must be greater than 0")
    loan_amount: float = Field(..., gt=0, description="Loan amount must be greater than 0")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score must be between 300 and 850")

class PredictionResponse(BaseModel):
    probability: float
    prediction: int

# --- API Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded or unhealthy."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(payload: CreditApplication):
    # Guard clause for model availability
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is currently unavailable."
        )
    
    try:
        # Convert Pydantic model to the format your model expects (e.g., Dict or DataFrame)
        input_data = [payload.dict()]
        
        # Make inference
        predictions = model.predict(input_data)
        
        # Assuming binary classification output; adapt based on your specific model return type
        prob = float(predictions[0]) if hasattr(predictions, "__iter__") else float(predictions)
        pred = 1 if prob >= 0.5 else 0
        
        return PredictionResponse(probability=prob, prediction=pred)
        
    except Exception as e:
        logger.error(f"Inference failure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during model inference."
        )