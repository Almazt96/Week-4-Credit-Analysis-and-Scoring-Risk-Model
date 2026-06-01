# Unit tests
import pytest
import pandas as pd
import numpy as np
from src.train import split_data, evaluate_model

@pytest.fixture
def sample_dataset():
    """Provides a synthetic DataFrame for processing tests."""
    np.random.seed(42)
    data = {
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
        "target": np.random.choice([0, 1], size=100)
    }
    return pd.DataFrame(data)

def test_split_data_shapes(sample_dataset):
    """Test 1: Verifies that data splitting maintains proper dimensions and ratios."""
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(sample_dataset, target_column="target", test_size=test_size)
    
    # Assert correct row lengths based on split 
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Assert column structures match expectations (target column dropped from features)
    assert "target" not in X_train.columns
    assert "feature_1" in X_train.columns

def test_evaluate_model_metrics():
    """Test 2: Assures performance metrics mapping logic outputs accurate figures."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])  # 4 matches out of 5, 1 false negative
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
    
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] == 0.80  # 4/5 correct assertions