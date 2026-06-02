import os
import joblib
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

# Global variable for the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles API startup and shutdown events cleanly.
    Loads the LightGBM champion model from the local container path.
    """
    global model
    try:
        logger.info("!!! TRACER: STARTING LOCAL MODEL LOAD PROCESS !!!")
        
        # Build path to champion_model.pkl sitting right next to main.py
        model_path = os.path.join(os.path.dirname(__file__), "champion_model.pkl")
        logger.info(f"Looking for model at local path: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}!")
            raise FileNotFoundError("Missing champion_model.pkl inside container.")

        # Load the local pkl file using joblib
        model = joblib.load(model_path)
        logger.info("🎉 SUCCESS: Champion model loaded beautifully from local file system!")
        
    except Exception as e:
        logger.critical(f"Critical error loading local model: {e}")
        # Crash the startup sequence explicitly if the model fails to load
        raise RuntimeError(f"Model initialization failed. API cannot start without a model. Internal error: {e}")
        
    yield
    logger.info("Shutting down API...")

# Initialize the one and ONLY app instance
app = FastAPI(
    title="Credit Risk Scoring API", 
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Request/Response Models ---
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
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is currently unavailable."
        )
    
    try:
        # Convert Pydantic model directly to a list of dicts for model input format
        input_data = [payload.dict()]
        
        # Make inference using your loaded joblib model
        predictions = model.predict(input_data)
        
        # Extract prediction probabilities (handling array structures safely)
        prob = float(predictions[0]) if hasattr(predictions, "__iter__") else float(predictions)
        pred = 1 if prob >= 0.5 else 0
        
        return PredictionResponse(probability=prob, prediction=pred)
        
    except Exception as e:
        logger.error(f"Inference failure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during model inference: {str(e)}"
        )