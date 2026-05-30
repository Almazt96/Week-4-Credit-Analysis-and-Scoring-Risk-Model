# Request/response schemas
from pydantic import BaseModel, Field
from typing import List

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