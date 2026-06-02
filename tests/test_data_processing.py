import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import your actual functions here
# from src.data_processing import aggregate_customer_data, engineer_proxy_target

# Mocking the implementation placeholders for illustration
def aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Example target function: collapses transactions to customer-level RFM."""
    reference_date = datetime(2026, 5, 1)
    
    # Simple logic mimicking what your code might do
    df['days_ago'] = (reference_date - pd.to_datetime(df['transaction_date'])).dt.days
    
    agg = df.groupby('customer_id').agg(
        recency=('days_ago', 'min'),
        frequency=('transaction_id', 'count'),
        monetary_value=('amount', 'sum')
    ).reset_index()
    return agg

def engineer_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """Example target function: sets proxy risk target based on RFM thresholds."""
    # Suppose high risk (1) means high recency AND low monetary value
    df['risk_target'] = np.where((df['recency'] > 30) & (df['monetary_value'] < 50), 1, 0)
    return df


# --- Pytest Fixtures ---

@pytest.fixture
def mock_transaction_data():
    """Generates a small transaction dataframe with deterministic outcomes."""
    today = datetime(2026, 5, 1)
    
    data = {
        'transaction_id': [1, 2, 3, 4, 5],
        'customer_id': ['CUST_A', 'CUST_A', 'CUST_B', 'CUST_C', 'CUST_C'],
        'transaction_date': [
            today - timedelta(days=5),   # CUST_A (Recent)
            today - timedelta(days=2),   # CUST_A (Recent)
            today - timedelta(days=45),  # CUST_B (Old / Inactive)
            today - timedelta(days=10),  # CUST_C (Moderate)
            today - timedelta(days=12)   # CUST_C (Moderate)
        ],
        'amount': [100.0, 150.0, 20.0, 200.0, 300.0]
    }
    return pd.DataFrame(data)


# --- Test Cases ---

def test_aggregate_customer_data(mock_transaction_data):
    """Verify that transaction history maps correctly to RFM metrics per customer."""
    
    processed_df = aggregate_customer_data(mock_transaction_data)
    
    # Assertions on shape and columns
    assert 'customer_id' in processed_df.columns
    assert len(processed_df) == 3  # Three unique customers: A, B, C
    
    # Set customer_id as index for simple lookups in assertions
    metrics = processed_df.set_index('customer_id')
    
    # Test Customer A (Frequent, high value, very recent)
    assert metrics.loc['CUST_A', 'frequency'] == 2
    assert metrics.loc['CUST_A', 'monetary_value'] == 250.0
    assert metrics.loc['CUST_A', 'recency'] == 2
    
    # Test Customer B (Single transaction, low value, old)
    assert metrics.loc['CUST_B', 'frequency'] == 1
    assert metrics.loc['CUST_B', 'recency'] == 45


def test_engineer_proxy_target():
    """Verify that the RFM rule engine flags risky customers accurately."""
    # Create an explicit customer-level dataframe to isolate the target logic
    customer_summary = pd.DataFrame({
        'customer_id': ['GOOD_1', 'RISKO_2'],
        'recency': [5, 50],             # Recent vs Old
        'frequency': [10, 1],
        'monetary_value': [500.0, 15.0] # High vs Low value
    })
    
    result_df = engineer_proxy_target(customer_summary)
    
    assert 'risk_target' in result_df.columns
    
    target_map = result_df.set_index('customer_id')['risk_target'].to_dict()
    
    # GOOD_1 shouldn't be flagged as risky (0)
    assert target_map['GOOD_1'] == 0
    # RISKO_2 meets the criteria for low engagement/high risk (1)
    assert target_map['RISKO_2'] == 1