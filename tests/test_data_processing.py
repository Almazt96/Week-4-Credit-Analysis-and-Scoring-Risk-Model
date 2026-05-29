# Unit tests
import pytest
import pandas as pd
import numpy as np

# Sample helper functions to mimic your data processing scripts
def feature_engineer_columns(df):
    """Adds a total_income column to the dataset."""
    df_copy = df.copy()
    if 'salary' in df_copy.columns and 'bonus' in df_copy.columns:
        df_copy['total_income'] = df_copy['salary'] + df_copy['bonus']
    return df_copy

def handle_missing_values(df, strategy="mean"):
    """Fills missing numeric values with the column mean."""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        if strategy == "mean":
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    return df_copy


# --- UNIT TESTS ---

def test_feature_engineer_columns_returns_expected_columns():
    """Test 1: Check if the feature engineering step successfully adds the new column."""
    # Arrange
    input_data = pd.DataFrame({
        'salary': [50000, 60000],
        'bonus': [5000, 4000]
    })
    
    # Act
    output_data = feature_engineer_columns(input_data)
    
    # Assert
    assert 'total_income' in output_data.columns
    assert output_data['total_income'].iloc[0] == 55000


def test_handle_missing_values_imputes_correctly():
    """Test 2: Check if missing values are correctly handled/filled."""
    # Arrange
    input_data = pd.DataFrame({
        'feature_a': [1.0, np.nan, 3.0]
    })
    
    # Act
    output_data = handle_missing_values(input_data, strategy="mean")
    
    # Assert
    assert output_data['feature_a'].isnull().sum() == 0
    assert output_data['feature_a'].iloc[1] == 2.0  # Mean of 1 and 3 is 2