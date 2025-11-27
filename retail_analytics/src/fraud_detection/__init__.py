"""
Fraud Detection Module
======================

Anomaly detection for fraudulent transaction identification using
Isolation Forest and other anomaly detection algorithms.
"""

from .feature_engineering import FraudFeatureEngineer
from .isolation_forest import IsolationForestDetector
from .model_evaluation import FraudEvaluator

__all__ = ["FraudFeatureEngineer", "IsolationForestDetector", "FraudEvaluator"]
