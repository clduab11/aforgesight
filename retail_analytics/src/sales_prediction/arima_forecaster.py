"""
ARIMA Forecasting Module
========================

Implements ARIMA and SARIMA models for time series forecasting
with automatic parameter selection and diagnostics.

Usage:
    from src.sales_prediction import ARIMAForecaster

    forecaster = ARIMAForecaster()
    forecaster.fit(df, 'date', 'sales')
    forecast = forecaster.predict(horizon=30)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from loguru import logger
import warnings
import itertools

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA/SARIMA forecasting with automatic parameter selection.

    Implements automated model selection using information criteria,
    seasonal decomposition, and comprehensive diagnostics.

    Example:
        >>> forecaster = ARIMAForecaster()
        >>> forecaster.fit(df, 'date', 'sales')
        >>> forecast = forecaster.predict(horizon=30)
        >>> diagnostics = forecaster.get_diagnostics()
    """

    def __init__(
        self,
        auto_order: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = True,
        seasonal_period: int = 12,
        information_criterion: str = 'aic'
    ):
        """
        Initialize ARIMA Forecaster.

        Args:
            auto_order: Automatically select optimal (p, d, q)
            max_p: Maximum AR order to test
            max_d: Maximum differencing order to test
            max_q: Maximum MA order to test
            seasonal: Include seasonal components
            seasonal_period: Seasonal period (e.g., 12 for monthly)
            information_criterion: Criterion for model selection ('aic', 'bic')
        """
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.information_criterion = information_criterion

        self.model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None
        self.train_data = None
        self.date_column = None
        self.value_column = None
        self.freq = None

        logger.info("ARIMAForecaster initialized")

    def fit(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to time series data.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            order: Manual (p, d, q) order (overrides auto)
            seasonal_order: Manual (P, D, Q, s) seasonal order

        Returns:
            Self for method chaining

        Example:
            >>> forecaster.fit(df, 'date', 'sales')
            >>> # Or with manual order
            >>> forecaster.fit(df, 'date', 'sales', order=(1, 1, 1))
        """
        self.date_column = date_column
        self.value_column = value_column

        # Prepare data
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Set index
        ts = df.set_index(date_column)[value_column]

        # Infer frequency
        self.freq = pd.infer_freq(ts.index)
        if self.freq is None:
            self.freq = 'D'
            logger.warning("Could not infer frequency, using daily")

        ts = ts.asfreq(self.freq)

        # Handle missing values
        if ts.isna().any():
            ts = ts.interpolate(method='time')

        self.train_data = ts

        # Determine order
        if order:
            self.order = order
        elif self.auto_order:
            self.order = self._find_optimal_order(ts)
        else:
            # Default order
            d = self._determine_differencing(ts)
            self.order = (1, d, 1)

        logger.info(f"Using ARIMA order: {self.order}")

        # Determine seasonal order
        if self.seasonal:
            if seasonal_order:
                self.seasonal_order = seasonal_order
            else:
                self.seasonal_order = self._find_seasonal_order(ts)
            logger.info(f"Using seasonal order: {self.seasonal_order}")

        # Fit model
        try:
            if self.seasonal and self.seasonal_order:
                self.model = SARIMAX(
                    ts,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(
                    ts,
                    order=self.order
                )

            self.fitted_model = self.model.fit(disp=False)
            logger.info(f"Model fitted. AIC: {self.fitted_model.aic:.2f}")

        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            # Fallback to simple model
            self.order = (1, 1, 1)
            self.seasonal_order = None
            self.model = ARIMA(ts, order=self.order)
            self.fitted_model = self.model.fit(disp=False)

        return self

    def predict(
        self,
        horizon: int = 30,
        confidence_interval: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate forecasts for future periods.

        Args:
            horizon: Number of periods to forecast
            confidence_interval: Confidence level for intervals

        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper

        Example:
            >>> forecast = forecaster.predict(horizon=30)
            >>> print(forecast[['ds', 'yhat']].head())
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate forecast
        forecast = self.fitted_model.get_forecast(steps=horizon)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=1 - confidence_interval)

        # Create future dates
        last_date = self.train_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq=self.freq
        )

        # Build result DataFrame
        result = pd.DataFrame({
            'ds': future_dates,
            'yhat': pred_mean.values,
            'yhat_lower': conf_int.iloc[:, 0].values,
            'yhat_upper': conf_int.iloc[:, 1].values
        })

        logger.info(f"Generated {horizon}-period forecast")
        return result

    def predict_in_sample(self) -> pd.DataFrame:
        """
        Get in-sample predictions for model evaluation.

        Returns:
            DataFrame with actual and predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.fitted_model.fittedvalues

        result = pd.DataFrame({
            'ds': self.train_data.index,
            'actual': self.train_data.values,
            'predicted': predictions.values
        })

        return result

    def _determine_differencing(self, ts: pd.Series) -> int:
        """Determine optimal differencing order using ADF test."""
        for d in range(self.max_d + 1):
            if d == 0:
                test_series = ts
            else:
                test_series = ts.diff(d).dropna()

            try:
                result = adfuller(test_series, autolag='AIC')
                p_value = result[1]

                if p_value < 0.05:
                    logger.debug(f"Series stationary at d={d} (p={p_value:.4f})")
                    return d
            except Exception:
                continue

        return 1  # Default

    def _find_optimal_order(self, ts: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using grid search."""
        d = self._determine_differencing(ts)

        best_score = float('inf')
        best_order = (1, d, 1)

        # Reduced search space for efficiency
        p_range = range(0, min(self.max_p + 1, 4))
        q_range = range(0, min(self.max_q + 1, 4))

        for p, q in itertools.product(p_range, q_range):
            if p == 0 and q == 0:
                continue

            try:
                model = ARIMA(ts, order=(p, d, q))
                fitted = model.fit(disp=False)

                if self.information_criterion == 'aic':
                    score = fitted.aic
                else:
                    score = fitted.bic

                if score < best_score:
                    best_score = score
                    best_order = (p, d, q)

            except Exception:
                continue

        logger.info(f"Optimal order: {best_order} ({self.information_criterion}={best_score:.2f})")
        return best_order

    def _find_seasonal_order(self, ts: pd.Series) -> Tuple[int, int, int, int]:
        """Find optimal seasonal order."""
        # Simplified seasonal order selection
        s = self.seasonal_period

        # Test a few common seasonal patterns
        candidates = [
            (1, 1, 1, s),
            (1, 0, 1, s),
            (0, 1, 1, s),
            (1, 1, 0, s)
        ]

        best_score = float('inf')
        best_order = (1, 1, 1, s)

        for seasonal_order in candidates:
            try:
                model = SARIMAX(
                    ts,
                    order=self.order or (1, 1, 1),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(disp=False, maxiter=50)

                if self.information_criterion == 'aic':
                    score = fitted.aic
                else:
                    score = fitted.bic

                if score < best_score:
                    best_score = score
                    best_order = seasonal_order

            except Exception:
                continue

        return best_order

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and statistics.

        Returns:
            Dictionary with diagnostic information

        Example:
            >>> diagnostics = forecaster.get_diagnostics()
            >>> print(f"AIC: {diagnostics['aic']}")
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        residuals = self.fitted_model.resid

        diagnostics = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'n_observations': len(self.train_data),
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'ljung_box_pvalue': self._ljung_box_test(residuals),
            'model_summary': str(self.fitted_model.summary())
        }

        return diagnostics

    def _ljung_box_test(self, residuals: pd.Series) -> float:
        """Perform Ljung-Box test on residuals."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=[10], return_df=True)
            return result['lb_pvalue'].values[0]
        except Exception:
            return None

    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """
        Plot model diagnostics including residuals and ACF/PACF.

        Args:
            save_path: Path to save the plot
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        fig = self.fitted_model.plot_diagnostics(figsize=(14, 10))

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved diagnostics plot: {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_forecast(
        self,
        forecast: pd.DataFrame,
        history_points: int = 100,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot historical data and forecast.

        Args:
            forecast: Forecast DataFrame from predict()
            history_points: Number of historical points to show
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Historical data
        history = self.train_data[-history_points:]
        ax.plot(history.index, history.values, label='Historical', linewidth=2)

        # Forecast
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast',
                linestyle='--', linewidth=2, color='orange')

        # Confidence interval
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            alpha=0.3,
            color='orange',
            label='95% CI'
        )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(self.value_column or 'Value', fontsize=12)
        ax.set_title('ARIMA Forecast', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved forecast plot: {save_path}")

        plt.tight_layout()
        plt.show()

    def cross_validate(
        self,
        n_splits: int = 5,
        test_size: int = 30
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.

        Args:
            n_splits: Number of train/test splits
            test_size: Size of each test set

        Returns:
            Dictionary with error metrics for each fold
        """
        if self.train_data is None:
            raise ValueError("No training data. Call fit() first.")

        ts = self.train_data
        total_size = len(ts)

        if total_size < test_size * (n_splits + 1):
            n_splits = max(1, total_size // test_size - 1)
            logger.warning(f"Reduced to {n_splits} splits due to data size")

        results = {'mape': [], 'rmse': [], 'mae': []}

        for i in range(n_splits):
            # Define train/test split
            test_end = total_size - i * test_size
            test_start = test_end - test_size
            train_end = test_start

            if train_end < test_size:
                break

            train = ts[:train_end]
            test = ts[test_start:test_end]

            try:
                # Fit model on training data
                if self.seasonal and self.seasonal_order:
                    model = SARIMAX(
                        train,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    model = ARIMA(train, order=self.order)

                fitted = model.fit(disp=False)

                # Forecast
                forecast = fitted.forecast(steps=len(test))

                # Calculate metrics
                actual = test.values
                predicted = forecast.values

                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mae = np.mean(np.abs(actual - predicted))

                results['mape'].append(mape)
                results['rmse'].append(rmse)
                results['mae'].append(mae)

            except Exception as e:
                logger.warning(f"CV fold {i} failed: {e}")

        # Calculate summary statistics
        for metric in results:
            if results[metric]:
                avg = np.mean(results[metric])
                std = np.std(results[metric])
                logger.info(f"CV {metric.upper()}: {avg:.2f} (+/- {std:.2f})")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        return {
            'type': 'ARIMA' if not self.seasonal else 'SARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.fitted_model.aic if self.fitted_model else None,
            'bic': self.fitted_model.bic if self.fitted_model else None,
            'n_observations': len(self.train_data) if self.train_data is not None else 0
        }
