import pytest
import pandas as pd
import numpy as np
from src.data_processing import build_feature_pipeline

@pytest.fixture
def sample_transaction_data():
    return pd.DataFrame({
        'CustomerId': ['C100', 'C100', 'C200'],
        'Amount': [150.0, 300.0, 45.0],
        'TransactionStartTime': ['2023-10-01 10:15:00', '2023-10-02 14:20:00', '2023-10-03 09:00:00'],
        'ProductId': ['P_ProdA', 'P_ProdA', 'P_ProdB'],
        'ProductCategory': ['Utility', 'Utility', 'Retail']
    })

def test_feature_pipeline_output_shape_and_columns(sample_transaction_data):
    pipeline = build_feature_pipeline()
    # Mock labels matching distinct groupings size structures
    y_mock = pd.Series([0, 1], index=['C100', 'C200'])
    
    X_out = pipeline.fit_transform(sample_transaction_data, y_mock)
    
    # Assert dimension structures contract boundaries validation
    assert isinstance(X_out, pd.DataFrame)
    assert X_out.shape[0] == 2  # Aggregated down to exactly 2 unique custom customers
    assert 'Total_Amount' in X_out.columns
    assert 'Transaction_Count' in X_out.columns
    assert 'TransactionHour' not in X_out.columns # verify nested mapping aggregated features properly down structural paths