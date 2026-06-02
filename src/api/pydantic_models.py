from pydantic import BaseModel, Field

class CreditApplication(BaseModel):
    # Enforce realistic and strict validation limits
    income: float = Field(..., gt=0, description="Income must be strictly positive.")
    loan_amount: float = Field(..., gt=0, description="Loan amount must be strictly positive.")
    credit_score: int = Field(..., ge=300, le=850, description="Valid US credit scores range from 300 to 850.")
    debt_to_income: float = Field(..., ge=0, le=2, description="Debt-to-income ratio expected between 0 and 2.0.")

    class Config:
        schema_extra = {
            "example": {
                "income": 55000.0,
                "loan_amount": 12000.0,
                "credit_score": 680,
                "debt_to_income": 0.28
            }
        }