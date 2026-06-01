# Feature engineering
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer, FunctionTransformer
from xverse.transformer import WOE

# 1. Custom Aggregator for Customer-Level Features
def aggregate_customer_data(df):
    agg_df = df.groupby('CustomerId').agg(
        TotalAmount=('TransactionAmount', 'sum'),
        AvgAmount=('TransactionAmount', 'mean'),
        VolAmount=('TransactionAmount', 'std'),
        TotalVolume=('TransactionAmount', 'count')
    ).fillna(0)
    return agg_df

# 2. Temporal Feature Extraction
def extract_temporal_features(df):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    return df.drop(columns=['TransactionStartTime'])

# 3. Pipeline Construction
def get_feature_pipeline():
    # Numeric preprocessing: Handle outliers and skew
    num_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ])
    
    # Categorical preprocessing: WoE Encoding
    # Note: Xverse handles IV filtering during fit
    cat_pipeline = Pipeline([
        ('woe', WOE(iv_thresholds={'min': 0.02, 'max': 0.5}))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, ['TotalAmount', 'AvgAmount', 'VolAmount', 'TotalVolume']),
        ('cat', cat_pipeline, ['ProductId', 'ProductCategory'])
    ])
    
    return preprocessor

# Example Workflow Usage
def process_data(raw_data):
    # Step 1: Temporal
    data = extract_temporal_features(raw_data)
    # Step 2: Aggregate
    cust_data = aggregate_customer_data(data)
    # Step 3: Transform
    pipeline = get_feature_pipeline()
    processed_features = pipeline.fit_transform(cust_data, y=cust_data['Target'])
    
    return processed_features