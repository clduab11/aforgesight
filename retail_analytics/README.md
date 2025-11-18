# Enterprise Retail Analytics Suite

An enterprise-grade, AI-powered suite of Python tools for retail analytics, featuring sales forecasting, customer segmentation, and fraud detection. Built with modularity, extensibility, and production-readiness in mind.

## Features

### 1. Sales Prediction (Time-Series Forecasting)
- **ARIMA/SARIMA**: Automatic parameter selection, seasonal decomposition
- **Prophet**: Multi-seasonality support, holiday effects, trend changepoints
- Model diagnostics, cross-validation, and comprehensive evaluation metrics
- Exportable forecasts with confidence intervals

### 2. Customer Segmentation (K-Means Clustering)
- **RFM Analysis**: Recency, Frequency, Monetary value calculations
- **K-Means++**: Automatic cluster selection via elbow/silhouette methods
- Feature engineering with behavioral signals
- Campaign recommendations per segment

### 3. Fraud Detection (Anomaly Detection)
- **Isolation Forest**: Unsupervised anomaly detection
- **Ensemble Methods**: LOF, One-Class SVM support
- Rich feature engineering for temporal/behavioral patterns
- Business-oriented metrics and threshold optimization

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd retail_analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python data/generate_sample_data.py
```

### Run Analytics

```bash
# Run all analytics
python run_analytics.py --task all

# Sales forecasting
python run_analytics.py --task forecast --data data/sample_sales.csv --model prophet --horizon 30

# Customer segmentation
python run_analytics.py --task segment --data data/sample_customers.csv --n-clusters 5

# Fraud detection
python run_analytics.py --task fraud --data data/sample_transactions.csv --contamination 0.01
```

## Project Structure

```
retail_analytics/
├── config/
│   └── settings.yaml          # Configuration settings
├── data/
│   ├── generate_sample_data.py
│   ├── sample_sales.csv
│   ├── sample_customers.csv
│   └── sample_transactions.csv
├── src/
│   ├── common/
│   │   ├── data_loader.py     # Data loading & validation
│   │   ├── preprocessing.py   # Data cleaning & transformation
│   │   ├── visualization.py   # Plotting utilities
│   │   └── reporting.py       # Report generation
│   ├── sales_prediction/
│   │   ├── arima_forecaster.py
│   │   ├── prophet_forecaster.py
│   │   └── model_evaluation.py
│   ├── customer_segmentation/
│   │   ├── rfm_features.py
│   │   ├── kmeans_clustering.py
│   │   └── segment_analysis.py
│   └── fraud_detection/
│       ├── feature_engineering.py
│       ├── isolation_forest.py
│       └── model_evaluation.py
├── api/
│   └── main.py                # FastAPI endpoints
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── outputs/                   # Generated reports & plots
├── run_analytics.py           # CLI runner
├── requirements.txt
└── README.md
```

## Detailed Usage

### Sales Forecasting

```python
from src.sales_prediction import ProphetForecaster, ForecastEvaluator
from src.common import DataLoader, Preprocessor

# Load and preprocess data
loader = DataLoader()
df = loader.load_csv('data/sample_sales.csv', date_columns=['date'])

preprocessor = Preprocessor()
ts_df = preprocessor.prepare_time_series(df, 'date', 'sales')

# Fit Prophet model
forecaster = ProphetForecaster(
    yearly_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode='multiplicative'
)
forecaster.fit(ts_df, 'date', 'sales')

# Generate forecast
forecast = forecaster.predict(horizon=30)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Evaluate model
evaluator = ForecastEvaluator()
in_sample = forecaster.predict_in_sample()
metrics = evaluator.calculate_metrics(in_sample['actual'], in_sample['predicted'])
print(f"MAPE: {metrics['mape']:.2f}%")
```

### Customer Segmentation

```python
from src.customer_segmentation import RFMFeatureEngineer, KMeansSegmenter
from src.common import DataLoader

# Load transaction data
loader = DataLoader()
df = loader.load_csv('data/sample_customers.csv', date_columns=['date'])

# Engineer RFM features
rfm_engineer = RFMFeatureEngineer()
rfm = rfm_engineer.calculate_rfm(df, 'customer_id', 'date', 'amount')
rfm = rfm_engineer.calculate_rfm_scores(rfm)
features = rfm_engineer.engineer_features(rfm, df, 'customer_id', 'date', 'amount')

# Segment customers
segmenter = KMeansSegmenter(n_clusters=None)  # Auto-select K
segmenter.fit(features, ['recency', 'frequency', 'monetary'])

# Get profiles and recommendations
profiles = segmenter.get_cluster_profiles(features)
recommendations = segmenter.get_campaign_recommendations(profiles)
for rec in recommendations:
    print(rec)
```

### Fraud Detection

```python
from src.fraud_detection import FraudFeatureEngineer, IsolationForestDetector
from src.common import DataLoader

# Load transaction data
loader = DataLoader()
df = loader.load_csv('data/sample_transactions.csv', date_columns=['timestamp'])

# Engineer features
engineer = FraudFeatureEngineer()
features = engineer.engineer_features(
    df, 'transaction_id', 'customer_id', 'amount', 'timestamp'
)

# Select feature columns
feature_cols = ['amount_log', 'hour', 'is_weekend', 'amount_zscore_global',
                'hours_since_last', 'amount_deviation', 'rapid_succession']

# Detect anomalies
detector = IsolationForestDetector(contamination=0.01)
detector.fit(features, feature_cols)

# Flag transactions
flagged = detector.flag_transactions(features)
print(f"Flagged {flagged['is_flagged'].sum()} suspicious transactions")
```

## API Usage

Start the API server:

```bash
uvicorn api.main:app --reload
```

### Endpoints

**POST /forecast**
```bash
curl -X POST "http://localhost:8000/forecast" \
  -F "file=@data/sample_sales.csv" \
  -F "horizon=30"
```

**POST /segment**
```bash
curl -X POST "http://localhost:8000/segment" \
  -F "file=@data/sample_customers.csv"
```

**POST /detect-fraud**
```bash
curl -X POST "http://localhost:8000/detect-fraud" \
  -F "file=@data/sample_transactions.csv" \
  -F "contamination=0.01"
```

## Docker Deployment

```bash
# Build and run all services
cd docker
docker-compose up -d

# Run analytics only
docker-compose up retail-analytics

# Run API server
docker-compose up retail-api

# Run Jupyter notebook
docker-compose up jupyter
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
sales_prediction:
  prophet:
    seasonality_mode: "multiplicative"
    yearly_seasonality: true
    weekly_seasonality: true

customer_segmentation:
  kmeans:
    n_clusters_range: [2, 10]
    init: "k-means++"

fraud_detection:
  isolation_forest:
    n_estimators: 100
    contamination: 0.01
```

## Data Requirements

### Sales Data
| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction date |
| sales | float | Sales amount |

### Customer Data
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string/int | Customer identifier |
| date | datetime | Transaction date |
| amount | float | Transaction amount |

### Transaction Data (Fraud)
| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string/int | Transaction identifier |
| customer_id | string/int | Customer identifier |
| timestamp | datetime | Transaction timestamp |
| amount | float | Transaction amount |

## Performance & Scaling

### Large Datasets (1M+ records)
- Use chunked data loading: `loader.load_csv(file, chunk_size=100000)`
- Enable parallel processing in scikit-learn models
- Use Parquet format for faster I/O: `loader.load_parquet(file)`

### Real-time Processing
- Deploy API with multiple workers: `uvicorn api.main:app --workers 4`
- Use Redis for caching model predictions
- Implement batch processing for fraud detection

## Key Metrics & Interpretation

### Forecast Metrics
- **MAPE < 10%**: Excellent accuracy
- **MAPE 10-20%**: Good accuracy
- **MAPE > 20%**: Consider model improvements

### Segmentation Metrics
- **Silhouette Score > 0.5**: Good cluster separation
- **Silhouette Score 0.25-0.5**: Moderate separation
- **Silhouette Score < 0.25**: Overlapping clusters

### Fraud Detection Metrics
- **Precision**: Proportion of flagged transactions that are actual fraud
- **Recall**: Proportion of actual frauds that were flagged
- **AUC-ROC > 0.9**: Excellent detection capability

## Extending the Suite

### Adding New Models

```python
# Create new forecaster in src/sales_prediction/
class LSTMForecaster:
    def __init__(self, config):
        self.model = None

    def fit(self, df, date_col, value_col):
        # Implementation
        pass

    def predict(self, horizon):
        # Implementation
        pass
```

### Custom Feature Engineering

```python
# Add to FraudFeatureEngineer
def _add_custom_features(self, df):
    df['custom_feature'] = # your logic
    return df
```

## Troubleshooting

### Common Issues

**Prophet installation fails**
```bash
pip install prophet --no-cache-dir
# Or use conda
conda install -c conda-forge prophet
```

**Memory errors with large datasets**
- Reduce feature dimensions
- Use sampling for initial analysis
- Enable chunked processing

**Poor model performance**
- Check data quality and missing values
- Tune hyperparameters
- Consider ensemble methods

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## Support

For issues and feature requests, please open a GitHub issue.

---

Built with best practices for enterprise retail analytics in 2025.
