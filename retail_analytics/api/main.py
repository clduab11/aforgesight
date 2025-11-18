"""
Retail Analytics API
====================

FastAPI endpoints for retail analytics services.

Usage:
    uvicorn api.main:app --reload

Endpoints:
    POST /forecast - Generate sales forecast
    POST /segment - Segment customers
    POST /detect-fraud - Detect fraudulent transactions
    GET /health - Health check
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from io import StringIO
from loguru import logger

from src.common import Preprocessor
from src.sales_prediction import ProphetForecaster, ForecastEvaluator
from src.customer_segmentation import RFMFeatureEngineer, KMeansSegmenter
from src.fraud_detection import FraudFeatureEngineer, IsolationForestDetector

# Initialize FastAPI
app = FastAPI(
    title="Retail Analytics API",
    description="AI-powered retail analytics services",
    version="1.0.0"
)

# Add CORS middleware with environment-based configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ForecastRequest(BaseModel):
    horizon: int = 30
    confidence_interval: float = 0.95


class SegmentRequest(BaseModel):
    n_clusters: Optional[int] = None
    features: List[str] = ["recency", "frequency", "monetary"]


class FraudRequest(BaseModel):
    contamination: float = 0.01
    threshold: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/forecast")
async def generate_forecast(
    file: UploadFile = File(...),
    request: ForecastRequest = ForecastRequest()
):
    """
    Generate sales forecast from uploaded CSV.

    Expected CSV format:
    - date: Date column
    - sales: Sales values
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Validate columns
        if 'date' not in df.columns or 'sales' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'date' and 'sales' columns"
            )

        # Preprocess
        preprocessor = Preprocessor()
        df['date'] = pd.to_datetime(df['date'])
        ts_df = preprocessor.prepare_time_series(df, 'date', 'sales')

        # Forecast
        forecaster = ProphetForecaster(
            interval_width=request.confidence_interval
        )
        forecaster.fit(ts_df, 'date', 'sales')
        forecast = forecaster.predict(horizon=request.horizon)

        # Evaluate
        evaluator = ForecastEvaluator()
        in_sample = forecaster.predict_in_sample()
        metrics = evaluator.calculate_metrics(
            in_sample['actual'].values,
            in_sample['predicted'].values
        )

        return {
            "status": "success",
            "forecast": forecast.to_dict('records'),
            "metrics": {
                "mape": round(metrics['mape'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mae": round(metrics['mae'], 2)
            },
            "model_info": forecaster.get_model_info()
        }

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
async def segment_customers(
    file: UploadFile = File(...),
    request: SegmentRequest = SegmentRequest()
):
    """
    Segment customers from uploaded transaction CSV.

    Expected CSV format:
    - customer_id: Customer identifier
    - date: Transaction date
    - amount: Transaction amount
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Validate columns
        required = ['customer_id', 'date', 'amount']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Preprocess
        preprocessor = Preprocessor()
        df = preprocessor.clean_data(df)
        df['date'] = pd.to_datetime(df['date'])

        # Engineer RFM features
        rfm_engineer = RFMFeatureEngineer()
        rfm = rfm_engineer.calculate_rfm(df, 'customer_id', 'date', 'amount')
        rfm = rfm_engineer.calculate_rfm_scores(rfm)
        features = rfm_engineer.engineer_features(rfm, df, 'customer_id', 'date', 'amount')

        # Cluster
        segmenter = KMeansSegmenter(n_clusters=request.n_clusters)
        segmenter.fit(features, request.features)

        features['cluster'] = segmenter.labels_
        profiles = segmenter.get_cluster_profiles(features)
        metrics = segmenter.get_cluster_metrics(features)
        recommendations = segmenter.get_campaign_recommendations(profiles)

        return {
            "status": "success",
            "n_clusters": segmenter.n_clusters,
            "n_customers": len(features),
            "cluster_profiles": profiles,
            "metrics": {
                "silhouette_score": round(metrics['silhouette_score'], 3),
                "inertia": round(metrics['inertia'], 2)
            },
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-fraud")
async def detect_fraud(
    file: UploadFile = File(...),
    request: FraudRequest = FraudRequest()
):
    """
    Detect fraudulent transactions from uploaded CSV.

    Expected CSV format:
    - transaction_id: Transaction identifier
    - customer_id: Customer identifier
    - timestamp: Transaction timestamp
    - amount: Transaction amount
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Validate columns
        required = ['transaction_id', 'customer_id', 'timestamp', 'amount']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Preprocess
        preprocessor = Preprocessor()
        df = preprocessor.clean_data(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Engineer features
        feature_engineer = FraudFeatureEngineer()
        features = feature_engineer.engineer_features(
            df, 'transaction_id', 'customer_id', 'amount', 'timestamp'
        )

        # Select feature columns
        exclude = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud',
                  'time_period', 'time_since_last']
        feature_cols = [c for c in features.columns
                       if c not in exclude
                       and features[c].dtype in ['int64', 'float64']]

        # Detect
        detector = IsolationForestDetector(contamination=request.contamination)
        detector.fit(features, feature_cols)
        flagged = detector.flag_transactions(features, threshold=request.threshold)

        # Get flagged transactions
        flagged_txns = flagged[flagged['is_flagged']][[
            'transaction_id', 'amount', 'anomaly_score', 'fraud_probability', 'risk_level'
        ]].to_dict('records')

        # Guard against division by zero
        flagged_rate = 0.0
        if len(flagged) > 0:
            flagged_rate = round(len(flagged_txns) / len(flagged) * 100, 2)

        return {
            "status": "success",
            "total_transactions": len(flagged),
            "flagged_count": len(flagged_txns),
            "flagged_rate": flagged_rate,
            "flagged_transactions": flagged_txns[:100],  # Limit to 100
            "threshold": detector.threshold,
            "recommendations": detector.get_recommendations(flagged)
        }

    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
