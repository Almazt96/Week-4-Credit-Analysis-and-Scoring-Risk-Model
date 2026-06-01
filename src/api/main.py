# FastAPI application
import os
import sys
import mlflow
import pandas as pd
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
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")
    
    try:
        # Convert Pydantic model to Pandas DataFrame for inference
        input_data = pd.DataFrame([payload.dict()])
        
        # Get probability output (assumes model outputs probability array or has a predict_proba equivalent wrapper)
        # Note: adjust inference signature based on how your model was logged (sklearn, xgboost, etc.)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            # Typically class 1 probability represents risk
            prob = float(probabilities[0][1])
        else:
            # Fallback if pyfunc outputs raw predictions/probabilities directly
            prediction = model.predict(input_data)
            prob = float(prediction[0])
            
        label = 1 if prob >= 0.5 else 0
        
        return PredictionResponse(risk_probability=prob, risk_label=label)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
    
    
    import pydantic_models
    from data_processing import create_full_processing_pipeline, ManualWoETransformer, TransactionAggregator