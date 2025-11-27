"""
Prophet Forecasting Module
==========================

Implements Facebook Prophet for time series forecasting with
custom seasonality, holidays, and trend change detection.

Usage:
    from src.sales_prediction import ProphetForecaster

    forecaster = ProphetForecaster()
    forecaster.fit(df, 'date', 'sales')
    forecast = forecaster.predict(horizon=30)
"""

import pandas as pd
from typing import Optional, Dict, List, Any
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Install with: pip install prophet")


class ProphetForecaster:
    """
    Facebook Prophet forecasting with automatic seasonality detection.

    Features:
    - Multiple seasonality support (yearly, weekly, daily)
    - Holiday effects modeling
    - Trend changepoint detection
    - Uncertainty quantification

    Example:
        >>> forecaster = ProphetForecaster()
        >>> forecaster.fit(df, 'date', 'sales')
        >>> forecast = forecaster.predict(horizon=30)
        >>> components = forecaster.get_components()
    """

    def __init__(
        self,
        growth: str = 'linear',
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.95
    ):
        """
        Initialize Prophet Forecaster.

        Args:
            growth: Growth model ('linear' or 'logistic')
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Enable yearly seasonality
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            holidays_prior_scale: Flexibility of holiday effects
            interval_width: Width of uncertainty intervals
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed")

        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.interval_width = interval_width

        self.model = None
        self.train_data = None
        self.forecast = None
        self.date_column = None
        self.value_column = None
        self.custom_seasonalities = []  # Store custom seasonalities to preserve across fit() calls
        self.custom_regressors = []  # Store custom regressors

        logger.info("ProphetForecaster initialized")

    def fit(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        holidays: Optional[pd.DataFrame] = None,
        regressors: Optional[List[str]] = None,
        cap: Optional[float] = None,
        floor: Optional[float] = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model to time series data.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            holidays: DataFrame with holiday dates and names
            regressors: List of additional regressor columns
            cap: Upper bound for logistic growth
            floor: Lower bound for logistic growth

        Returns:
            Self for method chaining

        Example:
            >>> forecaster.fit(df, 'date', 'sales')
            >>> # With holidays
            >>> holidays = pd.DataFrame({
            ...     'holiday': ['christmas', 'newyear'],
            ...     'ds': pd.to_datetime(['2024-12-25', '2025-01-01'])
            ... })
            >>> forecaster.fit(df, 'date', 'sales', holidays=holidays)
        """
        self.date_column = date_column
        self.value_column = value_column

        # Prepare data in Prophet format
        prophet_df = df[[date_column, value_column]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds')

        # Handle missing values
        if prophet_df['y'].isna().any():
            prophet_df['y'] = prophet_df['y'].interpolate(method='linear')

        # Add cap and floor for logistic growth
        if self.growth == 'logistic':
            if cap is None:
                cap = prophet_df['y'].max() * 1.5
            if floor is None:
                floor = 0
            prophet_df['cap'] = cap
            prophet_df['floor'] = floor

        self.train_data = prophet_df

        # Initialize Prophet model
        prophet_params = {
            "growth": self.growth,
            "seasonality_mode": self.seasonality_mode,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "interval_width": self.interval_width
        }
        if holidays is not None:
            prophet_params["holidays"] = holidays
        self.model = Prophet(**prophet_params)

        # Re-add any custom seasonalities that were previously configured
        for seasonality_config in self.custom_seasonalities:
            self.model.add_seasonality(**seasonality_config)
            logger.info(f"Re-added custom seasonality: {seasonality_config['name']}")

        # Add regressors
        if regressors:
            self.custom_regressors = regressors  # Store for later
            for regressor in regressors:
                self.model.add_regressor(regressor)
                prophet_df[regressor] = df[regressor].values

        # Fit model
        logger.info("Fitting Prophet model...")
        self.model.fit(prophet_df)
        logger.info(f"Model fitted on {len(prophet_df)} observations")

        return self

    def predict(
        self,
        horizon: int = 30,
        freq: str = 'D',
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Generate forecasts for future periods.

        Args:
            horizon: Number of periods to forecast
            freq: Frequency of predictions ('D', 'W', 'M', etc.)
            include_history: Include historical predictions

        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper

        Example:
            >>> forecast = forecaster.predict(horizon=30)
            >>> forecast = forecaster.predict(horizon=12, freq='M')
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq=freq,
            include_history=include_history
        )

        # Add cap/floor for logistic growth
        if self.growth == 'logistic':
            future['cap'] = self.train_data['cap'].iloc[0]
            future['floor'] = self.train_data['floor'].iloc[0]

        # Generate forecast
        self.forecast = self.model.predict(future)

        # Select relevant columns
        result = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

        logger.info(f"Generated {horizon}-period forecast")
        return result

    def predict_in_sample(self) -> pd.DataFrame:
        """
        Get in-sample predictions for model evaluation.

        Returns:
            DataFrame with actual and predicted values
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Predict on training data
        in_sample = self.model.predict(self.train_data)

        result = pd.DataFrame({
            'ds': self.train_data['ds'],
            'actual': self.train_data['y'],
            'predicted': in_sample['yhat']
        })

        return result

    def get_components(self) -> Dict[str, pd.DataFrame]:
        """
        Get forecast components (trend, seasonality).

        Returns:
            Dictionary with component DataFrames

        Example:
            >>> components = forecaster.get_components()
            >>> trend = components['trend']
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")

        components = {}

        # Trend
        components['trend'] = self.forecast[['ds', 'trend']].copy()

        # Yearly seasonality
        if 'yearly' in self.forecast.columns:
            components['yearly'] = self.forecast[['ds', 'yearly']].copy()

        # Weekly seasonality
        if 'weekly' in self.forecast.columns:
            components['weekly'] = self.forecast[['ds', 'weekly']].copy()

        # Daily seasonality
        if 'daily' in self.forecast.columns:
            components['daily'] = self.forecast[['ds', 'daily']].copy()

        # Holidays
        if 'holidays' in self.forecast.columns:
            components['holidays'] = self.forecast[['ds', 'holidays']].copy()

        return components

    def add_custom_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ) -> 'ProphetForecaster':
        """
        Add custom seasonality to the model.

        Must be called before fit().

        Args:
            name: Name of the seasonality component
            period: Period of the seasonality in days
            fourier_order: Number of Fourier terms
            prior_scale: Strength of the seasonality
            mode: 'additive' or 'multiplicative'

        Returns:
            Self for method chaining

        Example:
            >>> forecaster.add_custom_seasonality(
            ...     name='monthly',
            ...     period=30.5,
            ...     fourier_order=5
            ... )
        """
        if self.model is None:
            # Initialize model if not already done
            self.model = Prophet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale
            )

        kwargs = {'name': name, 'period': period, 'fourier_order': fourier_order}
        if prior_scale is not None:
            kwargs['prior_scale'] = prior_scale
        if mode is not None:
            kwargs['mode'] = mode

        # Store custom seasonality configuration
        self.custom_seasonalities.append(kwargs)
        
        # Add to model if it exists
        if self.model is not None:
            self.model.add_seasonality(**kwargs)
        
        logger.info(f"Added custom seasonality: {name} (period={period})")

        return self

    def cross_validate(
        self,
        initial: str = '730 days',
        period: str = '180 days',
        horizon: str = '365 days'
    ) -> pd.DataFrame:
        """
        Perform time series cross-validation.

        Args:
            initial: Initial training period
            period: Spacing between cutoff dates
            horizon: Forecast horizon

        Returns:
            DataFrame with cross-validation results

        Example:
            >>> cv_results = forecaster.cross_validate(
            ...     initial='365 days',
            ...     period='30 days',
            ...     horizon='30 days'
            ... )
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info("Running cross-validation...")

        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )

        # Calculate performance metrics
        metrics = performance_metrics(cv_results)

        logger.info(f"CV MAPE: {metrics['mape'].mean():.2%}")
        logger.info(f"CV RMSE: {metrics['rmse'].mean():.2f}")

        return cv_results

    def get_cv_metrics(
        self,
        cv_results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from cross-validation.

        Args:
            cv_results: Results from cross_validate()

        Returns:
            Dictionary of average metrics
        """
        metrics_df = performance_metrics(cv_results)

        metrics = {
            'mape': metrics_df['mape'].mean() * 100,
            'rmse': metrics_df['rmse'].mean(),
            'mae': metrics_df['mae'].mean(),
            'coverage': metrics_df['coverage'].mean() if 'coverage' in metrics_df else None
        }

        return metrics

    def plot_forecast(
        self,
        forecast: pd.DataFrame = None,
        save_path: Optional[str] = None
    ):
        """
        Plot forecast with Prophet's built-in plotting.

        Args:
            forecast: Forecast DataFrame (uses stored if None)
            save_path: Path to save the plot
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if forecast is None:
            if self.forecast is None:
                raise ValueError("No forecast available. Call predict() first.")
            forecast = self.forecast

        fig = self.model.plot(forecast)

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved forecast plot: {save_path}")

        return fig

    def plot_components(
        self,
        forecast: pd.DataFrame = None,
        save_path: Optional[str] = None
    ):
        """
        Plot forecast components (trend, seasonality).

        Args:
            forecast: Forecast DataFrame (uses stored if None)
            save_path: Path to save the plot
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if forecast is None:
            if self.forecast is None:
                raise ValueError("No forecast available. Call predict() first.")
            forecast = self.forecast

        fig = self.model.plot_components(forecast)

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved components plot: {save_path}")

        return fig

    def get_changepoints(self) -> pd.DataFrame:
        """
        Get detected trend changepoints.

        Returns:
            DataFrame with changepoint dates and deltas
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        changepoints = pd.DataFrame({
            'ds': self.model.changepoints,
            'delta': self.model.params['delta'].mean(axis=0)
        })

        return changepoints

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and statistics.

        Returns:
            Dictionary with diagnostic information
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        diagnostics = {
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'n_changepoints': len(self.model.changepoints),
            'n_observations': len(self.train_data),
            'components': list(self.model.seasonalities.keys())
        }

        return diagnostics

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        return {
            'type': 'Prophet',
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'interval_width': self.interval_width,
            'n_observations': len(self.train_data) if self.train_data is not None else 0,
            'confidence_interval': self.interval_width * 100
        }

    def save_model(self, path: str) -> None:
        """
        Save fitted model to disk.
        
        WARNING: Pickle files can execute arbitrary code during unpickling.
        Only load models from trusted sources.

        Args:
            path: Path to save model
        """
        import pickle

        # Save model and state
        model_state = {
            'model': self.model,
            'train_data': self.train_data,
            'forecast': self.forecast,
            'date_column': self.date_column,
            'value_column': self.value_column,
            'custom_seasonalities': self.custom_seasonalities,
            'custom_regressors': self.custom_regressors,
            'config': {
                'growth': self.growth,
                'seasonality_mode': self.seasonality_mode,
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'seasonality_prior_scale': self.seasonality_prior_scale,
                'holidays_prior_scale': self.holidays_prior_scale,
                'interval_width': self.interval_width
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> 'ProphetForecaster':
        """
        Load model from disk.
        
        WARNING: Pickle files can execute arbitrary code during unpickling.
        Only load models from trusted sources. Validate the file path before loading.

        Args:
            path: Path to load model from

        Returns:
            Self for method chaining
        """
        import pickle
        from pathlib import Path
        
        # Basic path validation
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if not model_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        # Restore state
        if isinstance(model_state, dict):
            # New format with full state
            self.model = model_state.get('model')
            self.train_data = model_state.get('train_data')
            self.forecast = model_state.get('forecast')
            self.date_column = model_state.get('date_column')
            self.value_column = model_state.get('value_column')
            self.custom_seasonalities = model_state.get('custom_seasonalities', [])
            self.custom_regressors = model_state.get('custom_regressors', [])
            
            # Restore config
            config = model_state.get('config', {})
            self.growth = config.get('growth', self.growth)
            self.seasonality_mode = config.get('seasonality_mode', self.seasonality_mode)
            self.yearly_seasonality = config.get('yearly_seasonality', self.yearly_seasonality)
            self.weekly_seasonality = config.get('weekly_seasonality', self.weekly_seasonality)
            self.daily_seasonality = config.get('daily_seasonality', self.daily_seasonality)
            self.changepoint_prior_scale = config.get('changepoint_prior_scale', self.changepoint_prior_scale)
            self.seasonality_prior_scale = config.get('seasonality_prior_scale', self.seasonality_prior_scale)
            self.holidays_prior_scale = config.get('holidays_prior_scale', self.holidays_prior_scale)
            self.interval_width = config.get('interval_width', self.interval_width)
        else:
            # Old format with just model
            self.model = model_state
            logger.warning("Loaded model in legacy format. Some state may not be restored.")

        logger.info(f"Model loaded from {path}")
        return self
