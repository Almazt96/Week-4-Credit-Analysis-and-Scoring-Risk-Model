# Request/response schemas
from pickle import GET
from venv import logger

from fastapi import FastAPI
from mlflow.server import health
from pydantic import BaseModel, Field
from typing import List

from uvicorn import logging

from src import predict

class CustomerData(BaseModel):
    # Adjust these fields to match your actual model's feature set
    age: int = Field(..., example=34, ge=18, le=100)
    income: float = Field(..., example=55000.0, ge=0)
    credit_score: int = Field(..., example=680, ge=300, le=850)
    employment_length: float = Field(..., example=4.5, ge=0)
    loan_amount: float = Field(..., example=15000.0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 34,
                "income": 55000.0,
                "credit_score": 680,
                "employment_length": 4.5,
                "loan_amount": 15000.0
            }
        }
    }

class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., example=0.23)
    risk_label: int = Field(..., example=0, description="0 for Low Risk, 1 for High Risk")
    prediction: str = Field(..., example="Low Risk", description="Human-readable prediction label")
    model_version: str = Field(..., example="1.0.0", description="Version of the model used for prediction")
    model_explanation: List[dict] = Field(..., example=[{"feature": "credit_score", "contribution": 0.15}, {"feature": "income", "contribution": 0.10}], description="List of feature contributions to the prediction")
    recommendation: str = Field(..., example="Consider improving credit score and increasing income to reduce risk.", description="Actionable recommendation based on the prediction")
    
class ErrorResponse(BaseModel):
    error_code: int = Field(..., example=400)
    error_message: str = Field(..., example="Invalid input data", description="Description of the error that occurred")
    
class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData] = Field(..., example=[{
    }    "age": 34,
        "income": 55000.0,
        "credit_score": 680,
        "employment_length": 4.5,
        "loan_amount": 15000.0
    }], description="List of customer data for batch prediction")
class healthCheckResponse(BaseModel):
    status: str = Field(..., example="OK", description="Health status of the API")
    
Logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@async def log_request(request):
    logging.info(f"Received request: {request.method} {request.url}")
    
@async def log_response(response):
    logging.info(f"Sending response: {response.status_code} {response.body}")
    
@asyccontextmanager
async def log_request_response(request, response):
    await log_request(request)
    yield
    await log_response(response)
    if response.status_code >= 400:
        logging.error(f"Error response: {response.status_code} - {response.body}")
    else:
        logging.info(f"Successful response: {response.status_code} - {response.body}")
    try:
        response_data = response.json()
        logging.info(f"Response data: {response_data}")
    except Exception as e:
        logging.error(f"Failed to parse response data: {e}")
    yield
    
    logger.info(f"Request: {request.method} {request.url} - Response: {response.status_code}")
    
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on customer data using a machine learning model.",
    version="1.0.0"
)

@app =middeleware(log_request_response)


What are the endpoints for the API?The API will have the following endpoints:
1. **POST /predict**: Accepts customer data and returns a credit risk prediction along with an explanation and recommendation.
2. **POST /batch_predict**: Accepts a list of customer data and returns predictions for each customer in the batch.
3. **GET /health**: Returns the health status of the API to confirm that it is running properly.    

Endpoint implementations will be defined in the main application file, where the logic for handling requests, making predictions using the machine learning model, and formatting responses will be implemented.

to check the health of the API, you can implement the following endpoint:
@app.get("/health", response_model=healthCheckResponse, summary="Health Check", description="Check the health status of the API")
async def health_check():    
    return healthCheckResponse(status="OK") 

predict endpoint implementation will be defined in the main application file, where the logic for handling requests, making predictions using the machine learning model, and formatting responses will be implemented.
