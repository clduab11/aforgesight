#!/usr/bin/env python3
"""
Retail Analytics Suite - Main Runner
=====================================

Command-line interface for running retail analytics pipelines.

Usage:
    python run_analytics.py --task forecast --data data/sample_sales.csv
    python run_analytics.py --task segment --data data/sample_customers.csv
    python run_analytics.py --task fraud --data data/sample_transactions.csv
    python run_analytics.py --task all --config config/settings.yaml

Examples:
    # Run sales forecasting with ARIMA
    python run_analytics.py --task forecast --model arima --horizon 30

    # Run customer segmentation with 5 clusters
    python run_analytics.py --task segment --n-clusters 5

    # Run fraud detection
    python run_analytics.py --task fraud --contamination 0.01
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.common import DataLoader, Preprocessor, Visualizer, Reporter
from src.sales_prediction import ARIMAForecaster, ProphetForecaster, ForecastEvaluator
from src.customer_segmentation import RFMFeatureEngineer, KMeansSegmenter, SegmentAnalyzer
from src.fraud_detection import FraudFeatureEngineer, IsolationForestDetector, FraudEvaluator


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def run_forecast(args, config):
    """Run sales forecasting pipeline."""
    logger.info("Starting Sales Forecasting Pipeline")

    # Load data
    loader = DataLoader()
    df = loader.load_csv(args.data, date_columns=['date'])

    # Preprocess
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)

    # Prepare time series
    ts_df = preprocessor.prepare_time_series(
        df, 'date', 'sales',
        freq='D', aggregate='sum'
    )

    # Select model
    if args.model == 'arima':
        logger.info("Using ARIMA model")
        forecaster = ARIMAForecaster(
            auto_order=True,
            seasonal=True,
            seasonal_period=7
        )
    else:  # prophet
        logger.info("Using Prophet model")
        forecaster = ProphetForecaster(
            yearly_seasonality=True,
            weekly_seasonality=True
        )

    # Fit model
    forecaster.fit(ts_df, 'date', 'sales')

    # Generate forecast
    forecast = forecaster.predict(horizon=args.horizon)

    # Evaluate
    evaluator = ForecastEvaluator()
    in_sample = forecaster.predict_in_sample()
    metrics = evaluator.calculate_metrics(
        in_sample['actual'].values,
        in_sample['predicted'].values
    )

    logger.info(f"Model Performance: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")

    # Visualize
    viz = Visualizer(output_dir=args.output)
    viz.plot_time_series(
        ts_df, 'date', 'sales',
        title='Sales Forecast',
        forecast_df=forecast.rename(columns={'ds': 'date', 'yhat': 'sales'}),
        confidence_interval=('yhat_lower', 'yhat_upper'),
        save_name='sales_forecast'
    )

    # Generate report
    reporter = Reporter(output_dir=args.output)
    results = {
        'forecast': forecast,
        'metrics': metrics,
        'model_info': forecaster.get_model_info()
    }
    reporter.generate_forecast_report(results, 'sales_forecast')

    logger.info(f"Forecast complete. Results saved to {args.output}")
    return results


def run_segmentation(args, config):
    """Run customer segmentation pipeline."""
    logger.info("Starting Customer Segmentation Pipeline")

    # Load data
    loader = DataLoader()
    df = loader.load_csv(args.data, date_columns=['date'])

    # Preprocess
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)

    # Engineer RFM features
    rfm_engineer = RFMFeatureEngineer()
    rfm = rfm_engineer.calculate_rfm(df, 'customer_id', 'date', 'amount')
    rfm = rfm_engineer.calculate_rfm_scores(rfm)

    # Engineer additional features
    features = rfm_engineer.engineer_features(
        rfm, df, 'customer_id', 'date', 'amount'
    )

    # Select features for clustering
    feature_cols = ['recency', 'frequency', 'monetary']

    # Cluster
    segmenter = KMeansSegmenter(
        n_clusters=args.n_clusters if args.n_clusters else None,
        random_state=42
    )
    segmenter.fit(features, feature_cols)

    # Get results
    features['cluster'] = segmenter.labels_
    profiles = segmenter.get_cluster_profiles(features)
    metrics = segmenter.get_cluster_metrics(features)

    # Get recommendations
    recommendations = segmenter.get_campaign_recommendations(profiles)

    logger.info(f"Segmentation complete. {segmenter.n_clusters} clusters identified.")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.3f}")

    # Visualize
    viz = Visualizer(output_dir=args.output)

    # Reduce dimensions for plotting
    reduced = segmenter.reduce_dimensions(features, method='pca')
    reduced['cluster'] = features['cluster']

    viz.plot_clusters(
        reduced, 'PC1', 'PC2', 'cluster',
        title='Customer Segments',
        save_name='customer_segments'
    )

    # Elbow plot
    plot_data = segmenter.get_selection_plot_data()
    viz.plot_elbow(
        plot_data['k_range'],
        plot_data['inertias'],
        plot_data['silhouettes'],
        save_name='cluster_selection'
    )

    # Generate report
    reporter = Reporter(output_dir=args.output)
    results = {
        'segments': features,
        'cluster_profiles': profiles,
        'metrics': metrics,
        'recommendations': recommendations
    }
    reporter.generate_segmentation_report(results, 'customer_segments')

    logger.info(f"Segmentation complete. Results saved to {args.output}")
    return results


def run_fraud_detection(args, config):
    """Run fraud detection pipeline."""
    logger.info("Starting Fraud Detection Pipeline")

    # Load data
    loader = DataLoader()
    df = loader.load_csv(args.data, date_columns=['timestamp'])

    # Preprocess
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)

    # Engineer features
    feature_engineer = FraudFeatureEngineer()
    features = feature_engineer.engineer_features(
        df,
        'transaction_id', 'customer_id', 'amount', 'timestamp'
    )

    # Select feature columns (exclude IDs and labels)
    exclude_cols = ['transaction_id', 'customer_id', 'timestamp', 'is_fraud',
                   'time_period', 'time_since_last', 'risk_level']
    feature_cols = [c for c in features.columns
                   if c not in exclude_cols
                   and features[c].dtype in ['int64', 'float64']]

    # Detect anomalies
    detector = IsolationForestDetector(
        contamination=args.contamination,
        n_estimators=100
    )
    detector.fit(features, feature_cols)

    # Flag transactions
    flagged = detector.flag_transactions(features)

    # Evaluate if labels available
    metrics = {}
    if 'is_fraud' in df.columns:
        evaluator = FraudEvaluator()
        y_true = df['is_fraud'].values
        y_pred = flagged['is_flagged'].astype(int).values
        y_scores = flagged['anomaly_score'].values

        metrics = evaluator.evaluate(y_true, y_pred, y_scores)
        logger.info(f"Detection Performance: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

        if metrics.get('auc_roc'):
            logger.info(f"AUC-ROC: {metrics['auc_roc']:.3f}")

    # Get recommendations
    recommendations = detector.get_recommendations(flagged)

    # Visualize
    viz = Visualizer(output_dir=args.output)
    viz.plot_anomaly_scores(
        flagged, 'anomaly_score',
        detector.threshold,
        date_column='timestamp',
        title='Anomaly Scores',
        save_name='anomaly_scores'
    )

    # Plot distribution
    viz.plot_distribution(
        flagged, 'anomaly_score',
        title='Anomaly Score Distribution',
        save_name='score_distribution'
    )

    # Generate report
    reporter = Reporter(output_dir=args.output)
    results = {
        'flagged_transactions': flagged[flagged['is_flagged']],
        'total_transactions': len(flagged),
        'metrics': metrics,
        'model_info': detector.get_model_info(),
        'recommendations': recommendations
    }
    reporter.generate_fraud_report(results, 'fraud_detection')

    n_flagged = flagged['is_flagged'].sum()
    logger.info(f"Fraud detection complete. Flagged {n_flagged} transactions ({n_flagged/len(flagged)*100:.2f}%)")
    logger.info(f"Results saved to {args.output}")

    return results


def run_all(args, config):
    """Run all analytics pipelines."""
    logger.info("Running Complete Analytics Suite")

    all_results = {}

    # Check for data files
    data_dir = Path('data')

    # Forecast
    sales_file = data_dir / 'sample_sales.csv'
    if sales_file.exists():
        args.data = str(sales_file)
        args.model = 'prophet'
        args.horizon = 30
        all_results['forecast'] = run_forecast(args, config)
    else:
        logger.warning(f"Sales data not found: {sales_file}")

    # Segmentation
    customers_file = data_dir / 'sample_customers.csv'
    if customers_file.exists():
        args.data = str(customers_file)
        args.n_clusters = None
        all_results['segmentation'] = run_segmentation(args, config)
    else:
        logger.warning(f"Customer data not found: {customers_file}")

    # Fraud detection
    transactions_file = data_dir / 'sample_transactions.csv'
    if transactions_file.exists():
        args.data = str(transactions_file)
        args.contamination = 0.01
        all_results['fraud'] = run_fraud_detection(args, config)
    else:
        logger.warning(f"Transaction data not found: {transactions_file}")

    # Generate executive summary
    reporter = Reporter(output_dir=args.output)
    reporter.generate_executive_summary(all_results)

    logger.info("Complete analytics suite finished")
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Retail Analytics Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--task',
        choices=['forecast', 'segment', 'fraud', 'all'],
        required=True,
        help='Analytics task to run'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to input data file'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    # Forecast options
    parser.add_argument(
        '--model',
        type=str,
        choices=['arima', 'prophet'],
        default='prophet',
        help='Forecasting model to use'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=30,
        help='Forecast horizon (number of periods)'
    )

    # Segmentation options
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (auto if not specified)'
    )

    # Fraud detection options
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        help='Expected fraud rate'
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    config = load_config(args.config)

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run task
    if args.task == 'forecast':
        if not args.data:
            parser.error("--data required for forecast task")
        run_forecast(args, config)

    elif args.task == 'segment':
        if not args.data:
            parser.error("--data required for segment task")
        run_segmentation(args, config)

    elif args.task == 'fraud':
        if not args.data:
            parser.error("--data required for fraud task")
        run_fraud_detection(args, config)

    elif args.task == 'all':
        run_all(args, config)


if __name__ == '__main__':
    main()
