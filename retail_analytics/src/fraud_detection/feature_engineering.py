"""
Fraud Feature Engineering Module
================================

Engineers features for fraud detection including temporal patterns,
behavioral signals, and statistical anomaly indicators.

Usage:
    from src.fraud_detection import FraudFeatureEngineer

    engineer = FraudFeatureEngineer()
    features = engineer.engineer_features(transactions)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class FraudFeatureEngineer:
    """
    Feature engineering for fraud detection in transactions.

    Creates features based on:
    - Transaction patterns (amount, frequency)
    - Temporal patterns (time of day, day of week)
    - Behavioral patterns (velocity, deviation from norm)
    - Statistical indicators (z-scores, percentiles)

    Example:
        >>> engineer = FraudFeatureEngineer()
        >>> features = engineer.engineer_features(transactions)
    """

    def __init__(
        self,
        time_windows: List[int] = [1, 24, 168],  # hours
        velocity_windows: List[int] = [1, 6, 24]  # hours
    ):
        """
        Initialize Fraud Feature Engineer.

        Args:
            time_windows: Time windows for aggregations (in hours)
            velocity_windows: Windows for velocity calculations (in hours)
        """
        self.time_windows = time_windows
        self.velocity_windows = velocity_windows
        self.customer_baselines = None

        logger.info("FraudFeatureEngineer initialized")

    def engineer_features(
        self,
        df: pd.DataFrame,
        transaction_id: str = 'transaction_id',
        customer_id: str = 'customer_id',
        amount_column: str = 'amount',
        timestamp_column: str = 'timestamp',
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features for fraud detection.

        Args:
            df: Transaction DataFrame
            transaction_id: Transaction ID column
            customer_id: Customer ID column
            amount_column: Transaction amount column
            timestamp_column: Timestamp column
            categorical_columns: Additional categorical columns

        Returns:
            DataFrame with engineered features

        Example:
            >>> features = engineer.engineer_features(
            ...     transactions, 'txn_id', 'cust_id', 'amount', 'timestamp'
            ... )
        """
        df = df.copy()

        # Ensure timestamp is datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column)

        logger.info(f"Engineering features for {len(df)} transactions")

        # Basic transaction features
        df = self._add_amount_features(df, amount_column)

        # Temporal features
        df = self._add_temporal_features(df, timestamp_column)

        # Customer behavior features
        df = self._add_customer_features(
            df, customer_id, amount_column, timestamp_column
        )

        # Velocity features
        df = self._add_velocity_features(
            df, customer_id, amount_column, timestamp_column
        )

        # Statistical anomaly features
        df = self._add_statistical_features(
            df, customer_id, amount_column
        )

        # Aggregation features
        df = self._add_aggregation_features(
            df, customer_id, amount_column, timestamp_column
        )

        # Encode categorical features
        if categorical_columns:
            df = self._encode_categorical(df, categorical_columns)

        logger.info(f"Engineered {len(df.columns)} features")
        return df

    def _add_amount_features(
        self,
        df: pd.DataFrame,
        amount_column: str
    ) -> pd.DataFrame:
        """Add amount-based features."""
        # Log transform
        df['amount_log'] = np.log1p(df[amount_column])

        # Binned amount
        df['amount_bin'] = pd.qcut(
            df[amount_column],
            q=10,
            labels=False,
            duplicates='drop'
        )

        # Amount characteristics
        df['is_round_amount'] = (df[amount_column] % 10 == 0).astype(int)
        df['is_large_amount'] = (
            df[amount_column] > df[amount_column].quantile(0.95)
        ).astype(int)

        # Digit features
        df['amount_cents'] = (df[amount_column] * 100 % 100).astype(int)
        df['ends_in_99'] = (df['amount_cents'] == 99).astype(int)

        return df

    def _add_temporal_features(
        self,
        df: pd.DataFrame,
        timestamp_column: str
    ) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df[timestamp_column].dt.hour
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        df['day_of_month'] = df[timestamp_column].dt.day
        df['month'] = df[timestamp_column].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Time period categorization
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )

        # Business hours
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)
        ).astype(int)

        # Unusual time (late night)
        df['is_unusual_time'] = (
            (df['hour'] >= 0) & (df['hour'] <= 5)
        ).astype(int)

        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_customer_features(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str,
        timestamp_column: str
    ) -> pd.DataFrame:
        """Add customer behavior features."""
        # Customer statistics
        customer_stats = df.groupby(customer_id).agg({
            amount_column: ['mean', 'std', 'min', 'max', 'count'],
            timestamp_column: ['min', 'max']
        }).reset_index()

        customer_stats.columns = [
            customer_id, 'cust_amount_mean', 'cust_amount_std',
            'cust_amount_min', 'cust_amount_max', 'cust_transaction_count',
            'cust_first_txn', 'cust_last_txn'
        ]

        # Account age in days
        customer_stats['cust_account_age'] = (
            customer_stats['cust_last_txn'] - customer_stats['cust_first_txn']
        ).dt.days

        # Average transactions per day
        customer_stats['cust_txn_per_day'] = (
            customer_stats['cust_transaction_count'] /
            (customer_stats['cust_account_age'] + 1)
        )

        # Merge back
        df = df.merge(
            customer_stats.drop(columns=['cust_first_txn', 'cust_last_txn']),
            on=customer_id,
            how='left'
        )

        # Deviation from customer mean
        df['amount_deviation'] = (
            df[amount_column] - df['cust_amount_mean']
        ) / (df['cust_amount_std'] + 1)

        # Is this customer's largest transaction?
        df['is_max_amount'] = (
            df[amount_column] == df['cust_amount_max']
        ).astype(int)

        # Ratio to average
        df['amount_to_avg_ratio'] = (
            df[amount_column] / (df['cust_amount_mean'] + 1)
        )

        return df

    def _add_velocity_features(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str,
        timestamp_column: str
    ) -> pd.DataFrame:
        """Add velocity and frequency features."""
        df = df.sort_values([customer_id, timestamp_column])

        # Time since last transaction
        df['time_since_last'] = df.groupby(customer_id)[timestamp_column].diff()
        df['hours_since_last'] = (
            df['time_since_last'].dt.total_seconds() / 3600
        ).fillna(0)

        # Amount change from last transaction
        df['amount_change'] = df.groupby(customer_id)[amount_column].diff().fillna(0)
        df['amount_change_pct'] = (
            df['amount_change'] /
            (df.groupby(customer_id)[amount_column].shift(1) + 1)
        ).fillna(0)

        # Transaction velocity (rolling count) - using vectorized operations
        # Sort by customer and timestamp for rolling operations
        df = df.sort_values([customer_id, timestamp_column])
        
        for window in self.velocity_windows:
            # Count transactions in window
            df[f'txn_count_{window}h'] = (
                df.groupby(customer_id, group_keys=False)
                .apply(lambda x: x.set_index(timestamp_column)[amount_column]
                       .rolling(f'{window}H').count())
            ).reset_index(level=0, drop=True)

            # Sum amount in window
            df[f'amount_sum_{window}h'] = (
                df.groupby(customer_id, group_keys=False)
                .apply(lambda x: x.set_index(timestamp_column)[amount_column]
                       .rolling(f'{window}H').sum())
            ).reset_index(level=0, drop=True)

        # Rapid succession indicator
        df['rapid_succession'] = (df['hours_since_last'] < 0.5).astype(int)

        return df

    def _add_statistical_features(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str
    ) -> pd.DataFrame:
        """Add statistical anomaly indicators."""
        # Global z-score
        global_mean = df[amount_column].mean()
        global_std = df[amount_column].std()
        df['amount_zscore_global'] = (
            (df[amount_column] - global_mean) / (global_std + 1e-10)
        )

        # Customer z-score (already have deviation)
        df['amount_zscore_customer'] = df['amount_deviation']

        # Percentile rank
        df['amount_percentile'] = df[amount_column].rank(pct=True)

        # Customer percentile rank
        df['amount_percentile_customer'] = df.groupby(customer_id)[amount_column].rank(pct=True)

        # Anomaly score (simple rule-based)
        df['high_zscore'] = (np.abs(df['amount_zscore_global']) > 3).astype(int)
        df['high_customer_zscore'] = (np.abs(df['amount_zscore_customer']) > 3).astype(int)

        return df

    def _add_aggregation_features(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str,
        timestamp_column: str
    ) -> pd.DataFrame:
        """Add time-window aggregation features."""
        df = df.sort_values([customer_id, timestamp_column])
        df = df.set_index(timestamp_column)

        for window in self.time_windows:
            window_str = f'{window}H'

            # Rolling statistics
            rolling = df.groupby(customer_id)[amount_column].rolling(window_str)

            df[f'amount_mean_{window}h'] = rolling.mean().values
            df[f'amount_std_{window}h'] = rolling.std().fillna(0).values
            df[f'amount_max_{window}h'] = rolling.max().values
            df[f'amount_min_{window}h'] = rolling.min().values

            # Count in window
            df[f'count_{window}h'] = rolling.count().values

        df = df.reset_index()
        return df

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Encode categorical features."""
        for col in columns:
            if col not in df.columns:
                continue

            # Frequency encoding
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)

            # Is rare category
            rare_threshold = 0.01
            rare_categories = freq[freq < rare_threshold].index
            df[f'{col}_is_rare'] = df[col].isin(rare_categories).astype(int)

        return df

    def calculate_customer_baselines(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str,
        timestamp_column: str
    ) -> pd.DataFrame:
        """
        Calculate baseline behavior for each customer.

        Args:
            df: Transaction DataFrame
            customer_id: Customer ID column
            amount_column: Amount column
            timestamp_column: Timestamp column

        Returns:
            DataFrame with customer baselines
        """
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        baselines = df.groupby(customer_id).agg({
            amount_column: ['mean', 'std', 'median', 'min', 'max'],
            timestamp_column: ['count', 'min', 'max']
        }).reset_index()

        baselines.columns = [
            customer_id,
            'baseline_mean', 'baseline_std', 'baseline_median',
            'baseline_min', 'baseline_max',
            'total_transactions', 'first_transaction', 'last_transaction'
        ]

        # Calculate typical patterns
        # Hour distribution
        hour_mode = df.groupby(customer_id)[timestamp_column].apply(
            lambda x: x.dt.hour.mode().iloc[0] if len(x) > 0 else 12
        ).reset_index()
        hour_mode.columns = [customer_id, 'typical_hour']

        baselines = baselines.merge(hour_mode, on=customer_id, how='left')

        self.customer_baselines = baselines
        logger.info(f"Calculated baselines for {len(baselines)} customers")

        return baselines

    def detect_baseline_deviations(
        self,
        df: pd.DataFrame,
        customer_id: str,
        amount_column: str,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect transactions that deviate from customer baselines.

        Args:
            df: Transaction DataFrame with features
            customer_id: Customer ID column
            amount_column: Amount column
            threshold: Z-score threshold for deviation

        Returns:
            DataFrame with deviation flags
        """
        if self.customer_baselines is None:
            raise ValueError("Baselines not calculated. Call calculate_customer_baselines() first.")

        df = df.merge(
            self.customer_baselines[[customer_id, 'baseline_mean', 'baseline_std']],
            on=customer_id,
            how='left'
        )

        # Calculate deviation
        df['baseline_deviation'] = (
            (df[amount_column] - df['baseline_mean']) /
            (df['baseline_std'] + 1)
        )

        df['exceeds_baseline'] = (
            np.abs(df['baseline_deviation']) > threshold
        ).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        base_features = [
            'amount_log', 'amount_bin', 'is_round_amount', 'is_large_amount',
            'amount_cents', 'ends_in_99',
            'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            'is_business_hours', 'is_unusual_time',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'cust_amount_mean', 'cust_amount_std', 'cust_transaction_count',
            'cust_account_age', 'cust_txn_per_day',
            'amount_deviation', 'is_max_amount', 'amount_to_avg_ratio',
            'hours_since_last', 'amount_change', 'amount_change_pct', 'rapid_succession',
            'amount_zscore_global', 'amount_zscore_customer',
            'amount_percentile', 'amount_percentile_customer',
            'high_zscore', 'high_customer_zscore'
        ]

        # Add velocity features
        for window in self.velocity_windows:
            base_features.extend([
                f'txn_count_{window}h',
                f'amount_sum_{window}h'
            ])

        # Add aggregation features
        for window in self.time_windows:
            base_features.extend([
                f'amount_mean_{window}h',
                f'amount_std_{window}h',
                f'amount_max_{window}h',
                f'amount_min_{window}h',
                f'count_{window}h'
            ])

        return base_features
