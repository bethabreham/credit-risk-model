"""
Feature Engineering Pipeline for Credit Risk Model
Transforms raw transaction data into model-ready features
Includes: RFM, time features, encoding, imputation, scaling, and proxy target
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


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
        exclude_cols = ['CustomerId']
        
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


def create_high_risk_label(df):
    """
    Create proxy target variable using RFM clustering
    High-risk = least engaged customers (low frequency, low monetary value)
    """
    
    print("=" * 60)
    print("PROXY TARGET VARIABLE ENGINEERING")
    print("=" * 60)
    
    # Select RFM columns for clustering
    rfm_cols = ['recency', 'frequency', 'monetary']
    
    # Check if columns exist
    available_cols = [col for col in rfm_cols if col in df.columns]
    if len(available_cols) < 3:
        # Use log-transformed versions if available
        if 'frequency_log' in df.columns and 'monetary_log' in df.columns:
            rfm_data = df[['recency', 'frequency_log', 'monetary_log']].copy()
            rfm_data.columns = ['recency', 'frequency', 'monetary']
        else:
            raise ValueError("RFM columns not found. Run feature engineering first.")
    else:
        rfm_data = df[available_cols].copy()
    
    print(f"\nRFM data shape: {rfm_data.shape}")
    print(f"RFM columns: {rfm_data.columns.tolist()}")
    
    # Handle infinite values
    rfm_data = rfm_data.replace([np.inf, -np.inf], np.nan)
    rfm_data = rfm_data.fillna(rfm_data.median())
    
    # Standardize RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # K-Means clustering
    print("\nPerforming K-Means clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    # Analyze clusters to identify high-risk group
    df_cluster = rfm_data.copy()
    df_cluster['cluster'] = clusters
    
    # Calculate cluster characteristics
    cluster_summary = df_cluster.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    }).round(2)
    
    print("\nCluster Characteristics:")
    print(cluster_summary)
    
    # Identify high-risk cluster (highest recency, lowest frequency, lowest monetary)
    cluster_scores = {}
    for c in cluster_summary.index:
        recency_score = cluster_summary.loc[c, 'recency']
        frequency_score = 1 / (cluster_summary.loc[c, 'frequency'] + 1)
        monetary_score = 1 / (cluster_summary.loc[c, 'monetary'] + 1)
        cluster_scores[c] = recency_score + frequency_score + monetary_score
    
    high_risk_cluster = max(cluster_scores, key=cluster_scores.get)
    
    print(f"\nHigh-risk cluster identified: Cluster {high_risk_cluster}")
    print(f"Characteristics: High recency, Low frequency, Low monetary value")
    
    # Create target variable
    df['is_high_risk'] = (clusters == high_risk_cluster).astype(int)
    
    print(f"\nTarget variable distribution:")
    print(df['is_high_risk'].value_counts())
    print(f"\nHigh-risk customers: {df['is_high_risk'].sum()} ({df['is_high_risk'].mean()*100:.2f}%)")
    
    return df


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
    print("\n✅ Customer features saved to 'data/processed/customer_features.csv'")
    
    # Create proxy target variable
    print("\n" + "=" * 60)
    print("CREATING PROXY TARGET VARIABLE")
    print("=" * 60)
    
    customer_features = create_high_risk_label(customer_features)
    customer_features.to_csv('data/processed/customer_features_with_target.csv', index=False)
    print("\n✅ Data with target saved to 'data/processed/customer_features_with_target.csv'")