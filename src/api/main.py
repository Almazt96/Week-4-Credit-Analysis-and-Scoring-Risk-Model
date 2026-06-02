# FastAPI application
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import mlflow
import pandas as pd
<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
=======
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerData, PredictionResponse

# Navigates 2 levels up from main.py to find the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Your original imports will now work perfectly:
from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI(
    title="FinTech Credit Risk Scoring API",
    description="Containerized API for predicting customer risk probabilities using MLflow models.",
    version="1.0.0"
)

# Load configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "best_credit_risk_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production") # Or use 'latest' / specific version URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Put anything you want to run on STARTUP here
    print("API is starting up... loading models...")
    
    yield  # The API runs and handles requests while paused here
    @app.on_event("startup")
    def load_model():
        global model
        try:
            # Construct the model URI from the registry
            model_uri = f"models://{MODEL_NAME}/{MODEL_STAGE}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Successfully loaded model: {MODEL_NAME} from stage: {MODEL_STAGE}")
        except Exception as e:
            print(f"Failed to load model from MLflow registry: {e}")
            # Fallback or initialization handling depending on local setup
            raise RuntimeError("Model initialization failed. API cannot start without a model.")
        
    # 2. Put anything you want to run on SHUTDOWN here (optional)
    print("API is shutting down... cleaning up...")

# Pass the lifespan handler to your FastAPI instance
app = FastAPI(lifespan=lifespan)

# ... rest of your endpoints (@app.get, @app.post, etc.)

@app.get("/")
def read_root():

from src.api.pydantic_models import PredictionRequest, PredictionResponse
>>>>>>> 484db9627391bacbc72e692f3bef176f281f48cf

# Global dictionary to hold the loaded model
ml_models: Dict[str, Any] = {}

<<<<<<< HEAD
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events cleanly.
    Loads the ML model into memory before the API starts receiving traffic.
    """
    try:
=======
# Load compiled workspace artifacts safely during initiation runtime
try:
    with open("models/pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("models/champion_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Fallback placeholders for deployment verification isolation runtimes
    pipeline, model = None, None

@app.get("/health")
def health_check():
  
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=list[PredictionResponse])
def predict_risk(payload: PredictionRequest):
    if not model or not pipeline:
        raise HTTPException(status_code=503, detail="Inference models not initialized.")
>>>>>>> 484db9627391bacbc72e692f3bef176f281f48cf
        
        # Force it to use a direct, local path instead of a models:/ registry URI
        # model_uri = r"D:\personal\kifiya 10 Academy\10 Academy\Week 4 credit-risk-model\mlruns\0\YOUR_ACTUAL_RUN_ID_HERE\artifacts\model"

        # Or if you found a different folder path:
        # model_uri = r"D:\path\to\your\saved\model_folder"
        # Inside src/api/main.py
        model_uri = "models:/Credit_Risk_Model/Production"
        ml_models["credit_risk_model"] = mlflow.pyfunc.load_model(model_uri)
        yield
    except Exception as e:
        print(f"Error during model loading: {str(e)}")
        # Prevent the server from starting in a broken state
        raise RuntimeError(f"Could not load model artifact: {e}")
    finally:
        # Clean up resources if necessary
        ml_models.clear()

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Schemas (v2 Syntax) ---

class CreditFeatureInput(BaseModel):
    """Input features required for credit risk prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    recency: int = Field(..., ge=0, description="Days since last transaction")
    frequency: int = Field(..., ge=0, description="Total number of transactions")
    monetary_value: float = Field(..., ge=0.0, description="Total monetary spend")
    # Add other specific features expected by your trained model pipeline here

class PredictionOutput(BaseModel):
    """Output structure returned to the client."""
    customer_id: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default/risk")
    prediction: int = Field(..., description="Binary classification outcome (0 or 1)")


# --- API Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Docker/K8s or CI/CD pinging."""
    if "credit_risk_model" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not loaded"
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict_risk(payload: CreditFeatureInput):
    """Accepts single-customer features and returns a risk score."""
    if "credit_risk_model" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model server is initializing or unhealthy."
        )
    
    try:
        # Convert input payload to a dictionary, discarding metadata if needed
        input_data = payload.model_dump()
        customer_id = input_data.pop("customer_id")
        
        # Convert to Pandas DataFrame as expected by typical MLflow pyfunc models
        input_df = pd.DataFrame([input_data])
        
        # Generate predictions
        model = ml_models["credit_risk_model"]
        
        # Adapt depending on if your model outputs probabilities directly or classes
        # Assuming typical sklearn-like pyfunc wrapper:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0][1])
            pred = int(prob >= 0.5)
        else:
            # Fallback if standard pyfunc predict returns array
            preds = model.predict(input_df)
            pred = int(preds[0])
            prob = float(pred)  # Adjust based on your model's exact signature
            
        return PredictionOutput(
            customer_id=customer_id,
            probability=prob,
            prediction=pred
        )
        
    except Exception as e:
<<<<<<< HEAD
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
        
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Credit Risk Model API",
        "status": "Healthy",
        "docs_url": "http://127.0.0.1:8000/docs"
    }
=======

        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
    
    
    import pydantic_models
    from data_processing import create_full_processing_pipeline, ManualWoETransformer, TransactionAggregator
        raise HTTPException(status_code=400, detail=f"Pipeline Processing Error: {str(e)}")
>>>>>>> 484db9627391bacbc72e692f3bef176f281f48cf
