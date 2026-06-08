"""
Save trained model for deployment
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier


def save_model():
    """Save the trained model and feature columns"""
    
    # Load processed data to get feature columns
    df = pd.read_csv('data/processed/customer_features_with_target.csv')
    
    # Drop CustomerId and target
    if 'CustomerId' in df.columns:
        df = df.drop('CustomerId', axis=1)
    
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']
    
    feature_columns = X.columns.tolist()
    print(f"Feature columns: {len(feature_columns)} features")
    
    # Train a Random Forest model (same as best model from Task 5)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X, y)
    print(f"Model trained. Accuracy: {model.score(X, y):.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved to models/random_forest_model.pkl")
    
    # Save feature columns
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print("✅ Feature columns saved to models/feature_columns.pkl")
    
    return model, feature_columns


if __name__ == "__main__":
    save_model()