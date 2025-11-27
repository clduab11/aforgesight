"""
Reporting Module
================

Generate comprehensive reports for retail analytics including
executive summaries, detailed metrics, and actionable insights.

Usage:
    from src.common import Reporter

    reporter = Reporter(output_dir="outputs/reports")
    reporter.generate_forecast_report(results, "Q4_2025_forecast")
    reporter.generate_executive_summary(all_results)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json
from loguru import logger


class Reporter:
    """
    Report generation for retail analytics results.

    Generates reports in multiple formats (CSV, JSON, HTML) with
    executive summaries and actionable insights.

    Example:
        >>> reporter = Reporter(output_dir="outputs/reports")
        >>> reporter.generate_forecast_report(forecast_results, "sales_forecast")
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize Reporter.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reporter initialized. Output: {self.output_dir}")

    def generate_forecast_report(
        self,
        results: Dict[str, Any],
        report_name: str,
        formats: List[str] = ['csv', 'json', 'html']
    ) -> Dict[str, Path]:
        """
        Generate comprehensive forecast report.

        Args:
            results: Forecast results dictionary containing:
                - forecast: DataFrame with predictions
                - metrics: Model performance metrics
                - model_info: Model configuration
            report_name: Base name for report files
            formats: Output formats to generate

        Returns:
            Dictionary of format -> file path

        Example:
            >>> results = {
            ...     'forecast': forecast_df,
            ...     'metrics': {'mape': 5.2, 'rmse': 120.5},
            ...     'model_info': {'type': 'Prophet', 'horizon': 30}
            ... }
            >>> paths = reporter.generate_forecast_report(results, "Q4_forecast")
        """
        output_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract components
        forecast = results.get('forecast', pd.DataFrame())
        metrics = results.get('metrics', {})
        model_info = results.get('model_info', {})

        # CSV Export
        if 'csv' in formats and not forecast.empty:
            csv_path = self.output_dir / f"{report_name}_{timestamp}.csv"
            forecast.to_csv(csv_path, index=False)
            output_paths['csv'] = csv_path
            logger.info(f"Saved CSV report: {csv_path}")

        # JSON Export
        if 'json' in formats:
            json_path = self.output_dir / f"{report_name}_{timestamp}.json"

            json_data = {
                'generated_at': timestamp,
                'model_info': model_info,
                'metrics': self._convert_to_serializable(metrics),
                'forecast_summary': {
                    'periods': len(forecast),
                    'start_date': str(forecast.iloc[0]['ds']) if 'ds' in forecast.columns and len(forecast) > 0 else None,
                    'end_date': str(forecast.iloc[-1]['ds']) if 'ds' in forecast.columns and len(forecast) > 0 else None,
                }
            }

            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            output_paths['json'] = json_path
            logger.info(f"Saved JSON report: {json_path}")

        # HTML Report
        if 'html' in formats:
            html_path = self.output_dir / f"{report_name}_{timestamp}.html"
            html_content = self._generate_forecast_html(forecast, metrics, model_info, report_name)

            with open(html_path, 'w') as f:
                f.write(html_content)
            output_paths['html'] = html_path
            logger.info(f"Saved HTML report: {html_path}")

        return output_paths

    def generate_segmentation_report(
        self,
        results: Dict[str, Any],
        report_name: str,
        formats: List[str] = ['csv', 'json', 'html']
    ) -> Dict[str, Path]:
        """
        Generate customer segmentation report.

        Args:
            results: Segmentation results containing:
                - segments: DataFrame with customer segments
                - cluster_profiles: Cluster statistics
                - metrics: Clustering metrics (silhouette, inertia)
                - recommendations: Campaign recommendations
            report_name: Base name for report files
            formats: Output formats

        Returns:
            Dictionary of format -> file path
        """
        output_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        segments = results.get('segments', pd.DataFrame())
        profiles = results.get('cluster_profiles', {})
        metrics = results.get('metrics', {})
        recommendations = results.get('recommendations', [])

        # CSV Export
        if 'csv' in formats and not segments.empty:
            csv_path = self.output_dir / f"{report_name}_{timestamp}.csv"
            segments.to_csv(csv_path, index=False)
            output_paths['csv'] = csv_path

        # JSON Export
        if 'json' in formats:
            json_path = self.output_dir / f"{report_name}_{timestamp}.json"

            json_data = {
                'generated_at': timestamp,
                'n_clusters': len(profiles),
                'metrics': self._convert_to_serializable(metrics),
                'cluster_profiles': self._convert_to_serializable(profiles),
                'recommendations': recommendations
            }

            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            output_paths['json'] = json_path

        # HTML Report
        if 'html' in formats:
            html_path = self.output_dir / f"{report_name}_{timestamp}.html"
            html_content = self._generate_segmentation_html(
                segments, profiles, metrics, recommendations, report_name
            )

            with open(html_path, 'w') as f:
                f.write(html_content)
            output_paths['html'] = html_path

        logger.info(f"Generated segmentation report: {report_name}")
        return output_paths

    def generate_fraud_report(
        self,
        results: Dict[str, Any],
        report_name: str,
        formats: List[str] = ['csv', 'json', 'html']
    ) -> Dict[str, Path]:
        """
        Generate fraud detection report.

        Args:
            results: Fraud detection results containing:
                - flagged_transactions: DataFrame of anomalies
                - metrics: Performance metrics
                - model_info: Model configuration
                - recommendations: Action recommendations
            report_name: Base name for report files
            formats: Output formats

        Returns:
            Dictionary of format -> file path
        """
        output_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        flagged = results.get('flagged_transactions', pd.DataFrame())
        metrics = results.get('metrics', {})
        model_info = results.get('model_info', {})
        recommendations = results.get('recommendations', [])

        # CSV Export
        if 'csv' in formats and not flagged.empty:
            csv_path = self.output_dir / f"{report_name}_{timestamp}.csv"
            flagged.to_csv(csv_path, index=False)
            output_paths['csv'] = csv_path

        # JSON Export
        if 'json' in formats:
            json_path = self.output_dir / f"{report_name}_{timestamp}.json"

            json_data = {
                'generated_at': timestamp,
                'model_info': model_info,
                'summary': {
                    'total_transactions': results.get('total_transactions', 0),
                    'flagged_count': len(flagged),
                    'flagged_percentage': len(flagged) / max(results.get('total_transactions', 1), 1) * 100
                },
                'metrics': self._convert_to_serializable(metrics),
                'recommendations': recommendations
            }

            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            output_paths['json'] = json_path

        # HTML Report
        if 'html' in formats:
            html_path = self.output_dir / f"{report_name}_{timestamp}.html"
            html_content = self._generate_fraud_html(
                flagged, metrics, model_info, recommendations, report_name
            )

            with open(html_path, 'w') as f:
                f.write(html_content)
            output_paths['html'] = html_path

        logger.info(f"Generated fraud detection report: {report_name}")
        return output_paths

    def generate_executive_summary(
        self,
        all_results: Dict[str, Dict[str, Any]],
        report_name: str = "executive_summary"
    ) -> Path:
        """
        Generate executive summary combining all analytics results.

        Args:
            all_results: Dictionary containing results from all modules:
                - 'forecast': Sales prediction results
                - 'segmentation': Customer segmentation results
                - 'fraud': Fraud detection results
            report_name: Base name for report file

        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"{report_name}_{timestamp}.html"

        html_content = self._generate_executive_html(all_results)

        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Generated executive summary: {html_path}")
        return html_path

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _generate_forecast_html(
        self,
        forecast: pd.DataFrame,
        metrics: Dict,
        model_info: Dict,
        report_name: str
    ) -> str:
        """Generate HTML report for forecast results."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_name} - Forecast Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .info-box {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sales Forecast Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Model Information</h2>
        <div class="info-box">
            <strong>Model Type:</strong> {model_info.get('type', 'N/A')}<br>
            <strong>Forecast Horizon:</strong> {model_info.get('horizon', 'N/A')} periods<br>
            <strong>Confidence Interval:</strong> {model_info.get('confidence_interval', 95)}%
        </div>

        <h2>Performance Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('mape', 'N/A'):.2f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('rmse', 'N/A'):.2f}</div>
                <div class="metric-label">RMSE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('mae', 'N/A'):.2f}</div>
                <div class="metric-label">MAE</div>
            </div>
        </div>

        <h2>Forecast Summary</h2>
        <table>
            <tr>
                <th>Period</th>
                <th>Forecast</th>
                <th>Lower Bound</th>
                <th>Upper Bound</th>
            </tr>
            {''.join(f"<tr><td>{row.get('ds', 'N/A')}</td><td>{row.get('yhat', 'N/A'):.2f}</td><td>{row.get('yhat_lower', 'N/A'):.2f}</td><td>{row.get('yhat_upper', 'N/A'):.2f}</td></tr>" for _, row in forecast.head(10).iterrows()) if not forecast.empty else '<tr><td colspan="4">No data</td></tr>'}
        </table>
        <p><em>Showing first 10 of {len(forecast)} periods</em></p>

        <h2>Key Insights</h2>
        <div class="info-box">
            <ul>
                <li>Model accuracy (MAPE): {metrics.get('mape', 'N/A'):.2f}% - {'Excellent' if metrics.get('mape', 100) < 10 else 'Good' if metrics.get('mape', 100) < 20 else 'Needs improvement'}</li>
                <li>Forecast trend: {'Upward' if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else 'Downward' if not forecast.empty else 'N/A'}</li>
                <li>Average forecasted value: {forecast['yhat'].mean():.2f if not forecast.empty else 'N/A'}</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    def _generate_segmentation_html(
        self,
        segments: pd.DataFrame,
        profiles: Dict,
        metrics: Dict,
        recommendations: List,
        report_name: str
    ) -> str:
        """Generate HTML report for segmentation results."""
        cluster_rows = ""
        for cluster_id, profile in profiles.items():
            cluster_rows += f"""
            <tr>
                <td>{cluster_id}</td>
                <td>{profile.get('size', 'N/A')}</td>
                <td>{profile.get('avg_recency', 'N/A'):.1f}</td>
                <td>{profile.get('avg_frequency', 'N/A'):.1f}</td>
                <td>${profile.get('avg_monetary', 'N/A'):.2f}</td>
            </tr>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_name} - Segmentation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #27ae60; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #27ae60; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .recommendation {{ background: #e8f8f0; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #27ae60; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Customer Segmentation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Clustering Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{len(profiles)}</div>
                <div class="metric-label">Clusters</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('silhouette_score', 'N/A'):.3f}</div>
                <div class="metric-label">Silhouette Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(segments)}</div>
                <div class="metric-label">Total Customers</div>
            </div>
        </div>

        <h2>Cluster Profiles</h2>
        <table>
            <tr>
                <th>Cluster</th>
                <th>Size</th>
                <th>Avg Recency (days)</th>
                <th>Avg Frequency</th>
                <th>Avg Monetary</th>
            </tr>
            {cluster_rows}
        </table>

        <h2>Campaign Recommendations</h2>
        {''.join(f'<div class="recommendation"><strong>Cluster {i}:</strong> {rec}</div>' for i, rec in enumerate(recommendations))}
    </div>
</body>
</html>
"""

    def _generate_fraud_html(
        self,
        flagged: pd.DataFrame,
        metrics: Dict,
        model_info: Dict,
        recommendations: List,
        report_name: str
    ) -> str:
        """Generate HTML report for fraud detection results."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_name} - Fraud Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #e74c3c; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #e74c3c; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .alert {{ background: #fdf2f2; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #e74c3c; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fraud Detection Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Detection Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{len(flagged)}</div>
                <div class="metric-label">Flagged Transactions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 'N/A'):.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 'N/A'):.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('auc_roc', 'N/A'):.3f}</div>
                <div class="metric-label">AUC-ROC</div>
            </div>
        </div>

        <h2>Model Configuration</h2>
        <div class="alert">
            <strong>Model:</strong> {model_info.get('type', 'Isolation Forest')}<br>
            <strong>Contamination:</strong> {model_info.get('contamination', 0.01)}<br>
            <strong>Estimators:</strong> {model_info.get('n_estimators', 100)}
        </div>

        <h2>Flagged Transactions (Top 10)</h2>
        <table>
            <tr>
                <th>Transaction ID</th>
                <th>Amount</th>
                <th>Anomaly Score</th>
                <th>Risk Level</th>
            </tr>
            {''.join(f"<tr><td>{row.get('transaction_id', 'N/A')}</td><td>${row.get('amount', 0):.2f}</td><td>{row.get('anomaly_score', 0):.3f}</td><td>{'High' if row.get('anomaly_score', 0) < -0.5 else 'Medium'}</td></tr>" for _, row in flagged.head(10).iterrows()) if not flagged.empty else '<tr><td colspan="4">No flagged transactions</td></tr>'}
        </table>

        <h2>Recommendations</h2>
        {''.join(f'<div class="alert">{rec}</div>' for rec in recommendations)}
    </div>
</body>
</html>
"""

    def _generate_executive_html(self, all_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate executive summary HTML."""
        forecast_results = all_results.get('forecast', {})
        segmentation_results = all_results.get('segmentation', {})
        fraud_results = all_results.get('fraud', {})

        # Precompute safe display variables for KPIs
        mape = forecast_results.get('metrics', {}).get('mape')
        mape_display = f"{mape:.1f}%" if isinstance(mape, (int, float)) else "N/A"
        mape_detailed = f"{mape:.2f}%" if isinstance(mape, (int, float)) else "N/A"
        
        cluster_profiles = segmentation_results.get('cluster_profiles', {})
        cluster_count_display = str(len(cluster_profiles)) if cluster_profiles else "0"
        
        flagged_transactions = fraud_results.get('flagged_transactions', [])
        flagged_tx_display = str(len(flagged_transactions)) if flagged_transactions else "0"
        
        auc = fraud_results.get('metrics', {}).get('auc_roc')
        auc_display = f"{auc:.2f}" if isinstance(auc, (int, float)) else "N/A"
        
        silhouette = segmentation_results.get('metrics', {}).get('silhouette_score')
        silhouette_display = f"{silhouette:.3f}" if isinstance(silhouette, (int, float)) else "N/A"
        
        precision = fraud_results.get('metrics', {}).get('precision')
        precision_display = f"{precision:.1%}" if isinstance(precision, (int, float)) else "N/A"
        
        model_type = forecast_results.get('model_info', {}).get('type', 'forecasting')
        
        # Determine forecast accuracy description
        if isinstance(mape, (int, float)):
            if mape < 10:
                accuracy_desc = "high"
            elif mape < 20:
                accuracy_desc = "moderate"
            else:
                accuracy_desc = "room for improvement in"
        else:
            accuracy_desc = "undetermined"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Executive Summary - Retail Analytics</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px 10px 0 0; }}
        .content {{ background: white; padding: 30px; border-radius: 0 0 10px 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 30px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .kpi-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-top: 4px solid; }}
        .kpi-card.blue {{ border-color: #3498db; }}
        .kpi-card.green {{ border-color: #27ae60; }}
        .kpi-card.red {{ border-color: #e74c3c; }}
        .kpi-card.purple {{ border-color: #9b59b6; }}
        .kpi-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .kpi-label {{ color: #7f8c8d; margin-top: 5px; font-size: 0.9em; }}
        .insight-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }}
        .insight-box h3 {{ margin-top: 0; color: #34495e; }}
        .action-item {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .timestamp {{ opacity: 0.8; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Retail Analytics Executive Summary</h1>
            <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="content">
            <h2>Key Performance Indicators</h2>
            <div class="kpi-grid">
                <div class="kpi-card blue">
                    <div class="kpi-value">{mape_display}</div>
                    <div class="kpi-label">Forecast Accuracy (MAPE)</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-value">{cluster_count_display}</div>
                    <div class="kpi-label">Customer Segments</div>
                </div>
                <div class="kpi-card red">
                    <div class="kpi-value">{flagged_tx_display}</div>
                    <div class="kpi-label">Flagged Transactions</div>
                </div>
                <div class="kpi-card purple">
                    <div class="kpi-value">{auc_display}</div>
                    <div class="kpi-label">Fraud Detection AUC</div>
                </div>
            </div>

            <h2>Sales Forecast Insights</h2>
            <div class="insight-box">
                <h3>Forecast Performance</h3>
                <p>The {model_type} model achieved a MAPE of {mape_detailed},
                indicating {accuracy_desc} prediction accuracy.</p>
                <p><strong>Recommendation:</strong> Use forecasts for inventory planning and staffing optimization.</p>
            </div>

            <h2>Customer Segmentation Insights</h2>
            <div class="insight-box">
                <h3>Segment Analysis</h3>
                <p>Identified {cluster_count_display} distinct customer segments with silhouette score of {silhouette_display}.</p>
                <p><strong>Recommendation:</strong> Develop targeted campaigns for each segment to maximize ROI.</p>
            </div>

            <h2>Fraud Detection Insights</h2>
            <div class="insight-box">
                <h3>Anomaly Detection</h3>
                <p>Detected {flagged_tx_display} potentially fraudulent transactions with precision of {precision_display}.</p>
                <p><strong>Recommendation:</strong> Review flagged transactions and implement automated blocking rules.</p>
            </div>

            <h2>Priority Action Items</h2>
            <div class="action-item">
                <strong>1. Inventory Optimization:</strong> Align stock levels with forecasted demand to reduce carrying costs.
            </div>
            <div class="action-item">
                <strong>2. Personalized Marketing:</strong> Launch segment-specific campaigns for top customer clusters.
            </div>
            <div class="action-item">
                <strong>3. Fraud Prevention:</strong> Implement real-time scoring for high-risk transactions.
            </div>
            <div class="action-item">
                <strong>4. Data Quality:</strong> Address any data gaps identified during analysis to improve model performance.
            </div>
        </div>
    </div>
</body>
</html>
"""
