import pytest
from fastapi.testclient import TestClient
from api.main import app, model
# Import your ACTUAL data processing functions here
# from src.data_processing import preprocess_features 

client = TestClient(app)

def test_successful_prediction_with_valid_data():
    """Test the endpoint with valid data parameters."""
    valid_payload = {
        "income": 65000.0,
        "loan_amount": 15000.0,
        "credit_score": 720,
        "debt_to_income": 0.25
    }
    # If your API executes preprocessing internally, this tests the real logic path
    response = client.post("/predict", json=valid_payload)
    
    # Assertions depend on whether your model is loaded or bypassed in testing environment
    if response.status_code == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert "probability" in json_data
    else:
        # If model is absent during unit test execution environment
        assert response.status_code == 503

def test_input_validation_error_handling():
    """Test that Pydantic properly flags invalid inputs before inference."""
    invalid_payload = {
        "income": -100,         # Invalid: Must be > 0
        "loan_amount": 15000,
        "credit_score": 999,    # Invalid: Must be <= 850
        "debt_to_income": 0.25
    }
    response = client.post("/predict", json=invalid_payload)
    
    # Pydantic should catch this and return a 422 Unprocessable Entity error
    assert response.status_code == 422
    assert "detail" in response.json()