from pydantic import BaseModel
from typing import List, Any

class TransactionRow(BaseModel):
    CustomerId: str
    Amount: float
    TransactionStartTime: str
    ProductId: str
    ProductCategory: str

class PredictionRequest(BaseModel):
    transactions: List[TransactionRow]

class PredictionResponse(BaseModel):
    customer_id: str
    credit_risk_probability: float
    is_high_risk: int