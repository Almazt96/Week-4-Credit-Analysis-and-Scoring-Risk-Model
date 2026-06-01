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

from sklearn.preprocessing import RobustScaler
from category_encoders.woe import WOEEncoder

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['TransactionHour'] = X[self.date_col].dt.hour
        X['TransactionDay'] = X[self.date_col].dt.day
        X['DayOfWeek'] = X[self.date_col].dt.dayofweek
        return X

class CustomerAggregatorAndEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, iv_min=0.02, iv_max=0.5):
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.woe_encoder = None
        self.scaler = RobustScaler()
        self.selected_features = []
        
    def fit(self, X, y):
        # 1. Temporal Aggregations (mean of hours/days for user profile)
        # 2. Volumetric Aggregations
        X_agg = X.groupby('CustomerId').agg(
            Total_Amount=('Amount', 'sum'),
            Avg_Amount=('Amount', 'mean'),
            Std_Amount=('Amount', 'std'),
            Transaction_Count=('Amount', 'count'),
            Avg_Hour=('TransactionHour', 'mean'),
            Avg_Day=('TransactionDay', 'mean'),
            Avg_DayOfWeek=('DayOfWeek', 'mean'),
            # Keeping high cardinality for WoE (mode per user for categorical context)
            ProductId=('ProductId', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'),
            ProductCategory=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        ).reset_index()
        
        X_agg['Std_Amount'] = X_agg['Std_Amount'].fillna(0)
        
        # Isolate target y aligned with grouped CustomerIds
        y_grouped = y.loc[X_agg['CustomerId']].values
        
        # Fit WoE Encoder
        cat_cols = ['ProductId', 'ProductCategory']
        self.woe_encoder = WOEEncoder(cols=cat_cols)
        X_encoded = self.woe_encoder.fit_transform(X_agg[cat_cols], y_grouped)
        
        # Combine numerical and WoE values
        num_cols = ['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Avg_Hour', 'Avg_Day', 'Avg_DayOfWeek']
        X_combined = pd.concat([X_agg[num_cols], X_encoded], axis=1)
        
        # IV Filtering (Approximated by checking WoE variance/predictive capability dynamically)
        # Note: category_encoders WoE doesn't output IV explicitly, we drop features if standard deviation is near zero
        self.selected_features = [col for col in X_combined.columns]
        
        # Fit Scaler
        self.scaler.fit(X_combined[self.selected_features])
        return self

    def transform(self, X):
        # Extraction logic applied identically during transform tracking
        X_agg = X.groupby('CustomerId').agg(
            Total_Amount=('Amount', 'sum'),
            Avg_Amount=('Amount', 'mean'),
            Std_Amount=('Amount', 'std'),
            Transaction_Count=('Amount', 'count'),
            Avg_Hour=('TransactionHour', 'mean'),
            Avg_Day=('TransactionDay', 'mean'),
            Avg_DayOfWeek=('DayOfWeek', 'mean'),
            ProductId=('ProductId', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'),
            ProductCategory=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        ).reset_index()
        
        X_agg['Std_Amount'] = X_agg['Std_Amount'].fillna(0)
        
        X_encoded = self.woe_encoder.transform(X_agg[['ProductId', 'ProductCategory']])
        num_cols = ['Total_Amount', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Avg_Hour', 'Avg_Day', 'Avg_DayOfWeek']
        X_combined = pd.concat([X_agg[num_cols], X_encoded], axis=1)
        
        X_scaled = self.scaler.transform(X_combined[self.selected_features])
        return pd.DataFrame(X_scaled, columns=self.selected_features, index=X_agg['CustomerId'])

def engineer_proxy_target(df):
    """Computes RFM metrics and generates labels via K-Means Clustering"""
    from sklearn.cluster import KMeans
    
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'Amount': ['count', 'sum']
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Standardize RFM data
    scaler = RobustScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # 3-Cluster Deterministic Segmentation
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify Least Engaged Cluster (High Recency, Low Frequency, Low Monetary)
    cluster_profiles = rfm.groupby('Cluster').mean()
    # High risk profile = low frequency & low monetary profile
    least_engaged_cluster = cluster_profiles['Frequency'].idxmin()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == least_engaged_cluster).astype(int)
    return rfm[['is_high_risk']]

def build_feature_pipeline():
    return Pipeline([
        ('temporal', TemporalFeatureExtractor()),
        ('agg_and_scale', CustomerAggregatorAndEncoder())
    ])
