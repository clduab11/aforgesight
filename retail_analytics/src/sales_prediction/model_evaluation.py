"""
Forecast Model Evaluation Module
================================

Comprehensive evaluation metrics and diagnostics for
time series forecasting models.

Usage:
    from src.sales_prediction import ForecastEvaluator

    evaluator = ForecastEvaluator()
    metrics = evaluator.calculate_metrics(actual, predicted)
    evaluator.compare_models(model_results)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class ForecastEvaluator:
    """
    Comprehensive evaluation toolkit for forecast models.

    Provides metrics calculation, model comparison, and
    diagnostic analysis for time series forecasts.

    Example:
        >>> evaluator = ForecastEvaluator()
        >>> metrics = evaluator.calculate_metrics(actual, predicted)
        >>> print(f"MAPE: {metrics['mape']:.2f}%")
    """

    def __init__(self):
        """Initialize ForecastEvaluator."""
        logger.info("ForecastEvaluator initialized")

    def calculate_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        include_advanced: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecast evaluation metrics.

        Args:
            actual: Actual values
            predicted: Predicted values
            include_advanced: Include advanced metrics

        Returns:
            Dictionary of metric names and values

        Example:
            >>> metrics = evaluator.calculate_metrics(actual, predicted)
            >>> print(f"RMSE: {metrics['rmse']:.2f}")
        """
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()

        # Basic metrics
        metrics = {
            'mse': mean_squared_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mae': mean_absolute_error(actual, predicted),
            'r2': r2_score(actual, predicted)
        }

        # MAPE (avoiding division by zero)
        mask = actual != 0
        if mask.any():
            metrics['mape'] = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            metrics['mape'] = np.nan

        # sMAPE (symmetric)
        denominator = np.abs(actual) + np.abs(predicted)
        mask = denominator != 0
        if mask.any():
            metrics['smape'] = np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
        else:
            metrics['smape'] = np.nan

        # Advanced metrics
        if include_advanced:
            # Mean Absolute Scaled Error (MASE)
            metrics['mase'] = self._calculate_mase(actual, predicted)

            # Tracking Signal
            metrics['tracking_signal'] = self._calculate_tracking_signal(actual, predicted)

            # Forecast Bias
            metrics['bias'] = np.mean(predicted - actual)

            # Theil's U Statistic
            metrics['theils_u'] = self._calculate_theils_u(actual, predicted)

            # Coverage (within +/- 10%) - guard against division by zero
            tolerance = 0.1
            # Only calculate for non-zero actual values
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                within_tolerance = np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]) <= tolerance
                metrics['coverage_10pct'] = np.mean(within_tolerance) * 100
            else:
                metrics['coverage_10pct'] = 0.0

        return metrics

    def _calculate_mase(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        n = len(actual)
        if n <= 1:
            return np.nan

        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))

        # Calculate scaling factor (naive forecast MAE)
        naive_mae = np.mean(np.abs(actual[1:] - actual[:-1]))

        if naive_mae == 0:
            return np.nan

        return mae / naive_mae

    def _calculate_tracking_signal(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Tracking Signal (cumulative error / MAD)."""
        errors = actual - predicted
        cumulative_error = np.sum(errors)
        mad = np.mean(np.abs(errors))

        if mad == 0:
            return np.nan

        return cumulative_error / mad

    def _calculate_theils_u(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        n = len(actual)
        if n <= 1:
            return np.nan

        # Relative changes
        actual_change = actual[1:] / actual[:-1] - 1
        predicted_change = predicted[1:] / predicted[:-1] - 1

        # Mask for valid values
        mask = ~(np.isnan(actual_change) | np.isnan(predicted_change) |
                 np.isinf(actual_change) | np.isinf(predicted_change))

        if not mask.any():
            return np.nan

        numerator = np.sqrt(np.mean((actual_change[mask] - predicted_change[mask])**2))
        denominator = np.sqrt(np.mean(actual_change[mask]**2)) + np.sqrt(np.mean(predicted_change[mask]**2))

        if denominator == 0:
            return np.nan

        return numerator / denominator

    def evaluate_interval_coverage(
        self,
        actual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_coverage: float = 0.95
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage and width.

        Args:
            actual: Actual values
            lower: Lower bound of prediction interval
            upper: Upper bound of prediction interval
            target_coverage: Expected coverage (e.g., 0.95)

        Returns:
            Dictionary with coverage metrics
        """
        actual = np.array(actual).flatten()
        lower = np.array(lower).flatten()
        upper = np.array(upper).flatten()

        # Coverage
        within_interval = (actual >= lower) & (actual <= upper)
        actual_coverage = np.mean(within_interval)

        # Interval width
        interval_width = upper - lower
        mean_width = np.mean(interval_width)
        # Guard against division by zero
        relative_width = np.mean(interval_width / np.where(actual != 0, actual, 1))

        # Winkler Score (lower is better)
        alpha = 1 - target_coverage
        winkler_score = self._calculate_winkler_score(actual, lower, upper, alpha)

        return {
            'actual_coverage': actual_coverage * 100,
            'target_coverage': target_coverage * 100,
            'coverage_gap': (actual_coverage - target_coverage) * 100,
            'mean_interval_width': mean_width,
            'relative_width': relative_width * 100,
            'winkler_score': winkler_score
        }

    def _calculate_winkler_score(
        self,
        actual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float
    ) -> float:
        """Calculate Winkler Score for prediction intervals."""
        width = upper - lower
        below = actual < lower
        above = actual > upper

        score = width.copy()
        score[below] += (2/alpha) * (lower[below] - actual[below])
        score[above] += (2/alpha) * (actual[above] - upper[above])

        return np.mean(score)

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Compare multiple forecast models.

        Args:
            model_results: Dictionary of model_name -> {actual, predicted}

        Returns:
            DataFrame with model comparisons

        Example:
            >>> results = {
            ...     'ARIMA': {'actual': y, 'predicted': arima_pred},
            ...     'Prophet': {'actual': y, 'predicted': prophet_pred}
            ... }
            >>> comparison = evaluator.compare_models(results)
        """
        comparisons = []

        for model_name, results in model_results.items():
            actual = results['actual']
            predicted = results['predicted']

            metrics = self.calculate_metrics(actual, predicted)
            metrics['model'] = model_name
            comparisons.append(metrics)

        comparison_df = pd.DataFrame(comparisons)
        comparison_df = comparison_df.set_index('model')

        # Rank models
        comparison_df['mape_rank'] = comparison_df['mape'].rank()
        comparison_df['rmse_rank'] = comparison_df['rmse'].rank()
        comparison_df['overall_rank'] = (comparison_df['mape_rank'] + comparison_df['rmse_rank']) / 2

        comparison_df = comparison_df.sort_values('overall_rank')

        logger.info(f"Model comparison complete. Best model: {comparison_df.index[0]}")
        return comparison_df

    def diebold_mariano_test(
        self,
        actual: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        h: int = 1,
        criterion: str = 'MSE'
    ) -> Dict[str, float]:
        """
        Perform Diebold-Mariano test for comparing forecast accuracy.

        Args:
            actual: Actual values
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            h: Forecast horizon
            criterion: Loss criterion ('MSE' or 'MAE')

        Returns:
            Dictionary with test statistics and p-value

        Example:
            >>> result = evaluator.diebold_mariano_test(actual, pred_arima, pred_prophet)
            >>> if result['p_value'] < 0.05:
            ...     print("Models have significantly different accuracy")
        """
        actual = np.array(actual).flatten()
        pred1 = np.array(pred1).flatten()
        pred2 = np.array(pred2).flatten()

        # Calculate loss differentials
        if criterion == 'MSE':
            e1 = (actual - pred1)**2
            e2 = (actual - pred2)**2
        else:  # MAE
            e1 = np.abs(actual - pred1)
            e2 = np.abs(actual - pred2)

        d = e1 - e2
        n = len(d)

        # Calculate variance with Newey-West correction
        mean_d = np.mean(d)
        gamma_0 = np.var(d)

        # Autocovariances
        gamma_sum = 0
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += gamma_k

        var_d = (gamma_0 + 2 * gamma_sum) / n

        # DM statistic
        if var_d <= 0:
            return {'dm_statistic': np.nan, 'p_value': np.nan}

        dm_stat = mean_d / np.sqrt(var_d)

        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'model1_better': dm_stat < 0
        }

    def residual_diagnostics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform diagnostic tests on forecast residuals.

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary with diagnostic test results

        Example:
            >>> diagnostics = evaluator.residual_diagnostics(actual, predicted)
            >>> if diagnostics['ljung_box_pvalue'] < 0.05:
            ...     print("Residuals show autocorrelation")
        """
        residuals = np.array(actual) - np.array(predicted)

        diagnostics = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }

        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            diagnostics['jarque_bera_stat'] = jb_stat
            diagnostics['jarque_bera_pvalue'] = jb_pvalue
            diagnostics['is_normal'] = jb_pvalue > 0.05
        except Exception:
            diagnostics['jarque_bera_stat'] = np.nan
            diagnostics['jarque_bera_pvalue'] = np.nan

        # Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            diagnostics['ljung_box_stat'] = lb_result['lb_stat'].values[0]
            diagnostics['ljung_box_pvalue'] = lb_result['lb_pvalue'].values[0]
            diagnostics['has_autocorrelation'] = lb_result['lb_pvalue'].values[0] < 0.05
        except Exception:
            diagnostics['ljung_box_stat'] = np.nan
            diagnostics['ljung_box_pvalue'] = np.nan

        # Heteroscedasticity (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            import statsmodels.api as sm
            X = sm.add_constant(np.arange(len(residuals)))
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
            diagnostics['breusch_pagan_stat'] = bp_stat
            diagnostics['breusch_pagan_pvalue'] = bp_pvalue
            diagnostics['is_homoscedastic'] = bp_pvalue > 0.05
        except Exception:
            diagnostics['breusch_pagan_stat'] = np.nan
            diagnostics['breusch_pagan_pvalue'] = np.nan

        return diagnostics

    def forecast_value_added(
        self,
        actual: np.ndarray,
        model_pred: np.ndarray,
        naive_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate Forecast Value Added (FVA) vs naive benchmark.

        Args:
            actual: Actual values
            model_pred: Model predictions
            naive_pred: Naive benchmark predictions (lag-1 if None)

        Returns:
            Dictionary with FVA metrics

        Example:
            >>> fva = evaluator.forecast_value_added(actual, predicted)
            >>> if fva['fva_mape'] > 0:
            ...     print(f"Model adds {fva['fva_mape']:.1f}% value over naive")
        """
        actual = np.array(actual).flatten()
        model_pred = np.array(model_pred).flatten()

        # Create naive forecast if not provided
        if naive_pred is None:
            naive_pred = np.roll(actual, 1)
            naive_pred[0] = actual[0]
        else:
            naive_pred = np.array(naive_pred).flatten()

        # Calculate metrics for both
        model_metrics = self.calculate_metrics(actual, model_pred, include_advanced=False)
        naive_metrics = self.calculate_metrics(actual, naive_pred, include_advanced=False)

        # Calculate FVA - guard against division by zero
        fva_mape_pct = 0.0
        if naive_metrics['mape'] != 0:
            fva_mape_pct = ((naive_metrics['mape'] - model_metrics['mape']) / naive_metrics['mape']) * 100
        
        fva_rmse_pct = 0.0
        if naive_metrics['rmse'] != 0:
            fva_rmse_pct = ((naive_metrics['rmse'] - model_metrics['rmse']) / naive_metrics['rmse']) * 100
        
        fva = {
            'model_mape': model_metrics['mape'],
            'naive_mape': naive_metrics['mape'],
            'fva_mape': naive_metrics['mape'] - model_metrics['mape'],
            'fva_mape_pct': fva_mape_pct,
            'model_rmse': model_metrics['rmse'],
            'naive_rmse': naive_metrics['rmse'],
            'fva_rmse': naive_metrics['rmse'] - model_metrics['rmse'],
            'fva_rmse_pct': fva_rmse_pct
        }

        return fva

    def generate_evaluation_report(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            actual: Actual values
            predicted: Predicted values
            lower: Lower prediction interval
            upper: Upper prediction interval
            model_name: Name of the model

        Returns:
            Dictionary with complete evaluation report
        """
        report = {
            'model_name': model_name,
            'n_observations': len(actual),
            'metrics': self.calculate_metrics(actual, predicted),
            'residual_diagnostics': self.residual_diagnostics(actual, predicted),
            'forecast_value_added': self.forecast_value_added(actual, predicted)
        }

        # Add interval evaluation if provided
        if lower is not None and upper is not None:
            report['interval_evaluation'] = self.evaluate_interval_coverage(
                actual, lower, upper
            )

        return report
