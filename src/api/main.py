"""
FastAPI Credit Risk Prediction Service
Loads trained model and provides /predict endpoint
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.pydantic_models import CustomerFeatures, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting customer credit risk using RFM clustering and ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for model and scaler
model = None
scaler = None
feature_columns = None


def load_model():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns
    
    try:
        # Try to load from pickle file
        model_path = 'models/random_forest_model.pkl'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/feature_columns.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        else:
            # Train a simple model if no saved model exists
            logger.warning("No saved model found. Training a simple model...")
            train_fallback_model()
        
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
            logger.info(f"Feature columns loaded: {len(feature_columns)} features")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def train_fallback_model():
    """Train a fallback model if no saved model exists"""
    global model, feature_columns
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Define feature columns (from our feature engineering)
    feature_columns = [
        'recency', 'frequency', 'monetary', 'avg_transaction', 'std_transaction',
        'monetary_log', 'frequency_log', 'fraud_count', 'fraud_rate', 'total_transactions',
        'product_financial_services', 'product_airtime', 'product_utility_bill',
        'product_data_bundles', 'product_tv', 'channel_ChannelId_1', 'channel_ChannelId_2',
        'channel_ChannelId_3', 'channel_ChannelId_5'
    ]
    
    # Create a simple decision-based model
    # This is a fallback - in production, you would load a real trained model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Note: In a real deployment, you would load the actual trained model
    logger.info("Fallback model created. For production, load a real trained model.")
    
    # Save feature columns
    os.makedirs('models', exist_ok=True)
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)


def predict_risk(features_dict):
    """Make risk prediction for a single customer"""
    global model, feature_columns
    
    if model is None:
        load_model()
    
    # Convert features to DataFrame
    input_df = pd.DataFrame([features_dict])
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training
    input_df = input_df[feature_columns]
    
    # Make prediction
    probability = model.predict_proba(input_df)[0, 1]
    
    # Determine risk label
    if probability >= 0.5:
        risk_label = "High Risk"
        confidence = probability
    else:
        risk_label = "Low Risk"
        confidence = 1 - probability
    
    return probability, risk_label, confidence


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    logger.info("API started successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict credit risk for a customer",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """
    Predict credit risk for a single customer
    
    - **risk_probability**: Probability of being high-risk (0-1)
    - **risk_label**: 'High Risk' or 'Low Risk'
    - **confidence**: Prediction confidence
    """
    try:
        # Convert Pydantic model to dict
        features_dict = features.model_dump()
        
        # Make prediction
        probability, risk_label, confidence = predict_risk(features_dict)
        
        return PredictionResponse(
            risk_probability=probability,
            risk_label=risk_label,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)