# FastAPI application
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
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
        raise HTTPException(status_code=400, detail=f"Pipeline Processing Error: {str(e)}")