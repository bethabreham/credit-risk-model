"""
Feature Engineering Pipeline for Credit Risk Model
Transforms raw transaction data into model-ready features
Includes: RFM, time features, encoding, imputation, scaling
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Extract time-based features from transaction timestamp"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        X['hour'] = X['TransactionStartTime'].dt.hour
        X['day'] = X['TransactionStartTime'].dt.day
        X['month'] = X['TransactionStartTime'].dt.month
        X['year'] = X['TransactionStartTime'].dt.year
        X['dayofweek'] = X['TransactionStartTime'].dt.dayofweek
        
        return X


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate features for RFM analysis"""
    
    def __init__(self):
        self.snapshot_date = None
    
    def fit(self, X, y=None):
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        self.snapshot_date = X['TransactionStartTime'].max()
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # RFM metrics per customer
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std']
        }).fillna(0)
        
        rfm.columns = ['recency', 'frequency', 'monetary', 'avg_transaction', 'std_transaction']
        rfm['monetary_log'] = np.log1p(rfm['monetary'])
        rfm['frequency_log'] = np.log1p(rfm['frequency'])
        
        rfm = rfm.reset_index()
        
        return rfm


class FraudFeature(BaseEstimator, TransformerMixin):
    """Aggregate fraud-related features per customer"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        fraud_features = X.groupby('CustomerId').agg({
            'FraudResult': ['sum', 'mean', 'count']
        })
        fraud_features.columns = ['fraud_count', 'fraud_rate', 'total_transactions']
        fraud_features = fraud_features.reset_index()
        
        return fraud_features


class ProductFeatures(BaseEstimator, TransformerMixin):
    """One-hot encode product categories and aggregate per customer"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Get top product categories
        top_categories = X['ProductCategory'].value_counts().head(5).index.tolist()
        
        # Create binary flags for each top category
        for category in top_categories:
            X[f'product_{category}'] = (X['ProductCategory'] == category).astype(int)
        
        # Aggregate per customer
        product_cols = [col for col in X.columns if col.startswith('product_')]
        product_features = X.groupby('CustomerId')[product_cols].sum().reset_index()
        
        return product_features


class ChannelFeatures(BaseEstimator, TransformerMixin):
    """One-hot encode channels and aggregate per customer"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create binary flags for each channel
        channels = X['ChannelId'].unique()
        for channel in channels:
            X[f'channel_{channel}'] = (X['ChannelId'] == channel).astype(int)
        
        # Aggregate per customer
        channel_cols = [col for col in X.columns if col.startswith('channel_')]
        channel_features = X.groupby('CustomerId')[channel_cols].sum().reset_index()
        
        return channel_features


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Standardize numerical features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols = []
    
    def fit(self, X, y=None):
        # Identify columns to scale (exclude CustomerId and binary columns)
        exclude_cols = ['CustomerId', 'recency', 'frequency', 'monetary', 'avg_transaction', 
                        'std_transaction', 'monetary_log', 'frequency_log', 'fraud_count', 
                        'fraud_rate', 'total_transactions']
        
        binary_cols = [col for col in X.columns if col.startswith('product_') or col.startswith('channel_')]
        exclude_cols.extend(binary_cols)
        
        self.scale_cols = [col for col in X.columns if col not in exclude_cols and col != 'CustomerId']
        
        if self.scale_cols:
            self.scaler.fit(X[self.scale_cols])
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.scale_cols:
            X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        return X


def create_full_pipeline():
    """Create the complete feature engineering pipeline"""
    
    pipeline = Pipeline([
        ('time_features', TimeFeatures()),
        ('aggregate_features', AggregateFeatures()),
        ('fraud_features', FraudFeature()),
        ('product_features', ProductFeatures()),
        ('channel_features', ChannelFeatures()),
        ('scaler', FeatureScaler())
    ])
    
    return pipeline


def create_customer_features(df):
    """Create all customer-level features for modeling"""
    
    print("Step 1: Extracting time features...")
    time_features = TimeFeatures()
    df = time_features.transform(df)
    
    print("Step 2: Creating RFM features...")
    rfm_features = AggregateFeatures()
    rfm_features.fit(df)
    customer_df = rfm_features.transform(df)
    
    print("Step 3: Creating fraud features...")
    fraud_features = FraudFeature()
    fraud_df = fraud_features.transform(df)
    customer_df = customer_df.merge(fraud_df, on='CustomerId', how='left')
    
    print("Step 4: Creating product features...")
    product_features = ProductFeatures()
    product_df = product_features.transform(df)
    customer_df = customer_df.merge(product_df, on='CustomerId', how='left')
    
    print("Step 5: Creating channel features...")
    channel_features = ChannelFeatures()
    channel_df = channel_features.transform(df)
    customer_df = customer_df.merge(channel_df, on='CustomerId', how='left')
    
    # Handle any remaining NaN values
    print("Step 6: Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    numeric_cols = customer_df.select_dtypes(include=[np.number]).columns
    customer_df[numeric_cols] = imputer.fit_transform(customer_df[numeric_cols])
    
    print("Step 7: Scaling numerical features...")
    scaler = FeatureScaler()
    scaler.fit(customer_df)
    customer_df = scaler.transform(customer_df)
    
    return customer_df


if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT RISK MODEL - FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\nLoading raw data...")
    df = pd.read_csv('data/raw/data.csv')
    print(f"Raw data shape: {df.shape}")
    
    # Create customer features
    print("\n" + "=" * 60)
    print("CREATING CUSTOMER-LEVEL FEATURES")
    print("=" * 60)
    
    customer_features = create_customer_features(df)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Customer features shape: {customer_features.shape}")
    print(f"Number of customers: {len(customer_features)}")
    
    # Save processed data
    customer_features.to_csv('data/processed/customer_features.csv', index=False)
    print("\n Customer features saved to 'data/processed/customer_features.csv'")