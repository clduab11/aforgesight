"""
Enterprise Retail Analytics Suite
==================================

A comprehensive suite of AI-powered tools for retail analytics including:
- Sales Prediction (ARIMA, Prophet)
- Customer Segmentation (K-Means, RFM)
- Fraud Detection (Isolation Forest, One-Class SVM)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Retail Analytics Team"

from .common import DataLoader, Preprocessor, Visualizer, Reporter
from .sales_prediction import ARIMAForecaster, ProphetForecaster
from .customer_segmentation import RFMFeatureEngineer, KMeansSegmenter
from .fraud_detection import FraudFeatureEngineer, IsolationForestDetector

__all__ = [
    "DataLoader",
    "Preprocessor",
    "Visualizer",
    "Reporter",
    "ARIMAForecaster",
    "ProphetForecaster",
    "RFMFeatureEngineer",
    "KMeansSegmenter",
    "FraudFeatureEngineer",
    "IsolationForestDetector",
]
