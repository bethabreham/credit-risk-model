"""
Pydantic models for FastAPI request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerFeatures(BaseModel):
    """Input features for credit risk prediction"""
    
    recency: float = Field(..., description="Days since last transaction")
    frequency: float = Field(..., description="Number of transactions")
    monetary: float = Field(..., description="Total transaction amount")
    avg_transaction: float = Field(..., description="Average transaction amount")
    std_transaction: float = Field(0.0, description="Standard deviation of transaction amounts")
    monetary_log: float = Field(..., description="Log of monetary value")
    frequency_log: float = Field(..., description="Log of frequency")
    fraud_count: float = Field(0.0, description="Number of fraud transactions")
    fraud_rate: float = Field(0.0, description="Rate of fraud transactions")
    total_transactions: float = Field(..., description="Total number of transactions")
    product_financial_services: int = Field(0, description="Financial services product flag")
    product_airtime: int = Field(0, description="Airtime product flag")
    product_utility_bill: int = Field(0, description="Utility bill product flag")
    product_data_bundles: int = Field(0, description="Data bundles product flag")
    product_tv: int = Field(0, description="TV product flag")
    channel_ChannelId_1: int = Field(0, description="Channel 1 flag")
    channel_ChannelId_2: int = Field(0, description="Channel 2 flag")
    channel_ChannelId_3: int = Field(0, description="Channel 3 flag")
    channel_ChannelId_5: int = Field(0, description="Channel 5 flag")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recency": 30,
                "frequency": 10,
                "monetary": 5000,
                "avg_transaction": 500,
                "std_transaction": 100,
                "monetary_log": 8.5,
                "frequency_log": 2.3,
                "fraud_count": 0,
                "fraud_rate": 0,
                "total_transactions": 10,
                "product_financial_services": 1,
                "product_airtime": 0,
                "product_utility_bill": 0,
                "product_data_bundles": 0,
                "product_tv": 0,
                "channel_ChannelId_1": 1,
                "channel_ChannelId_2": 0,
                "channel_ChannelId_3": 0,
                "channel_ChannelId_5": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint"""
    
    risk_probability: float = Field(..., description="Probability of being high-risk (0-1)")
    risk_label: str = Field(..., description="Risk category: 'High Risk' or 'Low Risk'")
    confidence: float = Field(..., description="Prediction confidence score")