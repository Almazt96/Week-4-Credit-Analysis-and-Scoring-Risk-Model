# Feature engineering
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class TransactionAggregator(BaseEstimator, TransformerMixin):
    """
    Transforms transaction-level logs into customer-level analytical records
    by creating aggregate features and extracting date-time metrics.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Ensure correct datetime parsing
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # 1. Temporal Feature Extraction (At transaction level before grouping)
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        # 2. Creating Aggregate Customer Features
        # For categorical features, we take the dominant (mode) value per customer
        agg_funcs = {
            'Amount': ['sum', 'mean', 'count', 'std'],
            'Value': ['sum', 'mean', 'std'],
            'TransactionHour': 'mean',
            'TransactionDay': 'mean',
            'TransactionMonth': 'mean',
            'TransactionYear': 'first',
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'PricingStrategy': lambda x: x.mode()[0] if not x.mode().empty else 0
        }
        
        customer_df = df.groupby('CustomerId').agg(agg_funcs)
        
        # Flatten multi-level column names resulting from aggregation
        customer_df.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
            for col in customer_df.columns
        ]
        customer_df = customer_df.reset_index()
        
        # Fill standard deviation NaNs (caused by customers with exactly 1 transaction)
        customer_df['Amount_std'] = customer_df['Amount_std'].fillna(0.0)
        customer_df['Value_std'] = customer_df['Value_std'].fillna(0.0)
        
        return customer_df


class ManualWoETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) value-mapping safely across critical columns
    to adhere to traditional credit rating scorecard expectations.
    """
    def __init__(self, columns_to_woe=None):
        self.columns_to_woe = columns_to_woe if columns_to_woe else []
        self.woe_maps = {}

    def fit(self, X, y=None):
        # In a real setup with ground-truth targets (y), you compute real WoE logs here.
        # For the proxy setup, we initialize stable structural mapping placeholders.
        if y is not None and len(self.columns_to_woe) > 0:
            df = pd.DataFrame(X).copy()
            df['target'] = y
            for col in self.columns_to_woe:
                # Grouped distribution counts
                total_good = (df['target'] == 0).sum()
                total_bad = (df['target'] == 1).sum()
                
                # Fallback to prevent divide by zero
                total_good = total_good if total_good > 0 else 1
                total_bad = total_bad if total_bad > 0 else 1

                # Group by bins/categories
                stats = df.groupby(col)['target'].agg(['count', 'sum'])
                stats['good'] = stats['count'] - stats['sum']
                stats['bad'] = stats['sum']
                
                # Calculate WoE percentages
                stats['woe'] = np.log(
                    (stats['good'] / total_good + 1e-5) / 
                    (stats['bad'] / total_bad + 1e-5)
                )
                self.woe_maps[col] = stats['woe'].to_dict()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, mapping in self.woe_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0.0)
        return df


def create_full_processing_pipeline(numerical_cols, categorical_cols):
    """
    Builds a comprehensive scikit-learn preprocessing Pipeline layout.
    """
    # Numerical sub-pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical sub-pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine columns transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # Master pipeline chain
    master_pipeline = Pipeline(steps=[
        ('aggregator', TransactionAggregator()),
        ('woe_mapping', ManualWoETransformer(columns_to_woe=['ChannelId_<lambda>', 'ProductCategory_<lambda>'])),
        ('column_transform', preprocessor)
    ])

    return master_pipeline


# Self-contained testing execution block
if __name__ == "__main__":
    print("🔄 Generating sample data matrix to verify data preprocessing script pipeline...")
    
    # Mock data resembling raw Xente transaction logs
    raw_sample_data = pd.DataFrame({
        'TransactionId': [f'T{i}' for i in range(1, 6)],
        'CustomerId': ['C_001', 'C_002', 'C_001', 'C_003', 'C_002'],
        'Amount': [5000.0, -1200.0, 3000.0, 15000.0, 400.0],
        'Value': [5000.0, 1200.0, 3000.0, 15000.0, 400.0],
        'TransactionStartTime': [
            '2026-05-28 14:20:00', 
            '2026-05-28 15:30:00', 
            '2026-05-29 09:15:00', 
            '2026-05-30 22:11:00', 
            '2026-05-30 11:05:00'
        ],
        'ChannelId': ['web', 'Android', 'web', 'pay-later', 'Android'],
        'ProductCategory': ['Airtime', 'UtilityBill', 'Airtime', 'FinancialServices', 'UtilityBill'],
        'PricingStrategy': [2, 4, 2, 1, 4]
    })

    # Explicit column configurations following TransactionAggregator flattening output names
    num_features = [
        'Amount_sum', 'Amount_mean', 'Amount_count', 'Amount_std',
        'Value_sum', 'Value_mean', 'Value_std',
        'TransactionHour_mean', 'TransactionDay_mean', 'TransactionMonth_mean'
    ]
    cat_features = ['ChannelId_<lambda>', 'ProductCategory_<lambda>', 'PricingStrategy_<lambda>']

    # Initialize and fit
    pipeline = create_full_processing_pipeline(num_features, cat_features)
    processed_matrix = pipeline.fit_transform(raw_sample_data)
    
    print("✅ Preprocessing check succeeded! Model-Ready output shape matrix:", processed_matrix.shape)