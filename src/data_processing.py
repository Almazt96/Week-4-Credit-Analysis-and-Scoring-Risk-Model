# Feature engineering
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# ==========================================
# TASK 3: CUSTOM TRANSFORMERS
# ==========================================

class FeatureEngineerAndAggregator(BaseEstimator, TransformerMixin):
    """
    Extracts datetime features, calculates aggregate features per customer,
    and returns a customer-level aggregated dataframe.
    """
    def __init__(self, date_col='TransactionStartTime', customer_id_col='CustomerId', amount_col='Amount'):
        self.date_col = date_col
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Extract datetime components
        df['TransactionHour'] = df[self.date_col].dt.hour
        df['TransactionDay'] = df[self.date_col].dt.day
        df['TransactionMonth'] = df[self.date_col].dt.month
        df['TransactionYear'] = df[self.date_col].dt.year
        
        # Handle simple categorical mode per customer for baseline categorical tracking (e.g., ProductId, ChannelId)
        # For this script, we'll focus heavily on the requested structural aggregations
        agg_funcs = {
            self.amount_col: ['sum', 'mean', 'count', 'std'],
            'TransactionHour': 'mean',
            'TransactionDay': 'mean',
            'TransactionMonth': 'mean'
        }
        
        # Dynamic handling if other categorical columns are present
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.drop([self.customer_id_col], errors='ignore')
        for col in cat_cols:
            agg_funcs[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan

        # Aggregate per customer
        customer_df = df.groupby(self.customer_id_col).agg(agg_funcs)
        
        # Flatten MultiIndex columns
        customer_df.columns = [
            f"{col}_{stat}" if isinstance(stat, str) else col 
            for col, stat in customer_df.columns
        ]
        customer_df = customer_df.reset_index()
        
        # Rename structural columns to match objectives
        customer_df.rename(columns={
            f"{self.amount_col}_sum": "Total_Transaction_Amount",
            f"{self.amount_col}_mean": "Average_Transaction_Amount",
            f"{self.amount_col}_count": "Transaction_Count",
            f"{self.amount_col}_std": "Standard_Deviation_Amount"
        }, inplace=self)
        
        return customer_df


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) mapping manually to avoid rigid third-party library constraints,
    ensuring stability inside the pipeline.
    Note: Requires a target variable during fit. If target is missing (Task 3 phase), it passes through.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.woe_maps = {}

    def fit(self, X, y=None):
        if y is None or self.cat_cols is None:
            return self
        
        df = X.copy()
        df['target'] = y
        
        for col in self.cat_cols:
            if col in df.columns:
                # Calculate WoE: ln(% of Goods / % of Bads)
                total_pos = df['target'].sum()
                total_neg = len(df) - total_pos
                
                # Smooth to avoid division by zero
                stats = df.groupby(col)['target'].agg(['count', 'sum'])
                stats['bads'] = stats['sum']
                stats['goods'] = stats['count'] - stats['bads']
                
                stats['prop_goods'] = (stats['goods'] + 0.5) / total_neg
                stats['prop_bads'] = (stats['bads'] + 0.5) / total_pos
                
                self.woe_maps[col] = np.log(stats['prop_goods'] / stats['prop_bads']).to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        for col, woe_map in self.woe_maps.items():
            if col in df.columns:
                df[col] = df[col].map(woe_map).fillna(0) # Default to 0 (neutral WoE) if category unseen
        return df


# ==========================================
# TASK 4: CUSTOM PROXY TARGET TRANSFORMER
# ==========================================

class RFMTargetEngineer(BaseEstimator, TransformerMixin):
    """
    Calculates RFM metrics, clusters customers using KMeans, and creates the proxy target variable 'is_high_risk'.
    """
    def __init__(self, date_col='TransactionStartTime', customer_id_col='CustomerId', amount_col='Amount', random_state=42):
        self.date_col = date_col
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()
        self.high_risk_cluster_id = None

    def fit(self, X, y=None):
        # We need raw transaction log or intermediate df to compute Recency
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # 1. Define Snapshot Date (Max date + 1 day)
        snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        
        # 2. Calculate RFM Core metrics
        rfm = df.groupby(self.customer_id_col).agg({
            self.date_col: lambda x: (snapshot_date - x.max()).days, # Recency
            self.customer_id_col: 'count',                          # Frequency
            self.amount_col: 'sum'                                   # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # 3. Scale RFM for clustering
        scaled_rfm = self.scaler.fit_transform(rfm)
        
        # 4. Fit K-Means
        self.kmeans.fit(scaled_rfm)
        rfm['Cluster'] = self.kmeans.labels_
        
        # 5. Identify High-Risk Cluster (Low Frequency, Low Monetary, High Recency)
        # We look for the lowest Mean Frequency + Monetary cluster profile
        cluster_profiles = rfm.groupby('Cluster').mean()
        # Sorting by a combined profile index where high risk = low frequency and low monetary
        self.high_risk_cluster_id = cluster_profiles['Frequency'].idxmin()
        
        return self

    def transform(self, X):
        # Re-calculate RFM dynamically on incoming data to assign targets
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby(self.customer_id_col).agg({
            self.date_col: lambda x: (snapshot_date - x.max()).days,
            self.customer_id_col: 'count',
            self.amount_col: 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        scaled_rfm = self.scaler.transform(rfm)
        clusters = self.kmeans.predict(scaled_rfm)
        
        # Map High-Risk Cluster to 1, others to 0
        rfm['is_high_risk'] = [1 if c == self.high_risk_cluster_id else 0 for c in clusters]
        
        return rfm[['is_high_risk']].reset_index()


# ==========================================
# MASTER PIPELINE EXECUTION FUNCTION
# ==========================================

def build_and_run_pipeline(raw_data_path: str) -> pd.DataFrame:
    """
    Loads raw data, executes full feature engineering pipeline,
    generates proxy targets via RFM clustering, and outputs a model-ready dataframe.
    """
    # Load Raw Data
    df_raw = pd.read_csv(raw_data_path)
    
    # --- Step 1: Generate High-Risk Target Variables (Task 4 Blueprint) ---
    target_engineer = RFMTargetEngineer()
    target_df = target_engineer.fit_transform(df_raw)
    
    # --- Step 2: Extract & Aggregate Features (Task 3 Blueprint) ---
    feature_pipeline = Pipeline([
        ('aggregator', FeatureEngineerAndAggregator()),
    ])
    
    processed_customer_df = feature_pipeline.fit_transform(df_raw)
    
    # --- Step 3: Merge Target and Processed Features ---
    final_df = pd.merge(processed_customer_df, target_df, on='CustomerId', how='left')
    
    # --- Step 4: Final Cleansing/Standardization (Handling Missing values & scaling numericals) ---
    # Separate numeric columns (excluding IDs and targets)
    exclude_cols = ['CustomerId', 'is_high_risk']
    num_cols = final_df.select_dtypes(include=[np.number]).columns.drop(exclude_cols, errors='ignore')
    
    # Impute missing values (e.g., standard deviations that resulted in NaN due to 1 transaction)
    imputer = SimpleImputer(strategy='median')
    final_df[num_cols] = imputer.fit_transform(final_df[num_cols])
    
    # Standardize numerical metrics
    scaler = StandardScaler()
    final_df[num_cols] = scaler.fit_transform(final_df[num_cols])
    
    # --- Step 5: Post-Target Weight of Evidence (WoE) Engine ---
    # Dynamically extract tracking categories if available (e.g., ProductId if aggregated as mode)
    cat_cols = final_df.select_dtypes(include=['object']).columns.drop(['CustomerId'], errors='ignore')
    if len(cat_cols) > 0:
        woe = WoETransformer(cat_cols=list(cat_cols))
        final_df = woe.fit_transform(final_df, y=final_df['is_high_risk'])
        
    return final_df

if __name__ == "__main__":
    # Example execution script structure
    print("Pipeline script initialized successfully.")
    # To execute in production:
    # processed_data = build_and_run_pipeline('data/raw_transactions.csv')
    # processed_data.to_csv('data/model_ready_dataset.csv', index=False)