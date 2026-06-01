# FastAPI application
import pickle
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

from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Bati Bank Real-Time Credit Scoring Service")

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
        
    # Standardize input transaction matrices payload mapping blocks
    raw_data = [t.model_dump() for t in payload.transactions]
    df_input = pd.DataFrame(raw_data)
    
    try:
        # Pass input data through pipeline processing pipeline structures
        X_transformed = pipeline.transform(df_input)
        
        probabilities = model.predict_proba(X_transformed)[:, 1]
        predictions = model.predict(X_transformed)
        
        responses = []
        for idx, customer_id in enumerate(X_transformed.index):
            responses.append(PredictionResponse(
                customer_id=str(customer_id),
                credit_risk_probability=float(probabilities[idx]),
                is_high_risk=int(predictions[idx])
            ))
        return responses
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
    
    
    import pydantic_models
    from data_processing import create_full_processing_pipeline, ManualWoETransformer, TransactionAggregator
        raise HTTPException(status_code=400, detail=f"Pipeline Processing Error: {str(e)}")
