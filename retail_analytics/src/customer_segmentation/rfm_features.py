"""
RFM Feature Engineering Module
==============================

Calculates Recency, Frequency, and Monetary value features
for customer segmentation with additional behavioral signals.

Usage:
    from src.customer_segmentation import RFMFeatureEngineer

    engineer = RFMFeatureEngineer()
    rfm_df = engineer.calculate_rfm(transactions_df)
    features = engineer.engineer_features(rfm_df, customers_df)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class RFMFeatureEngineer:
    """
    RFM and behavioral feature engineering for customer analytics.

    Calculates Recency, Frequency, Monetary value and additional
    behavioral features for customer segmentation.

    Example:
        >>> engineer = RFMFeatureEngineer()
        >>> rfm = engineer.calculate_rfm(transactions, 'customer_id', 'date', 'amount')
        >>> features = engineer.engineer_features(rfm, customers)
    """

    def __init__(
        self,
        recency_bins: int = 5,
        frequency_bins: int = 5,
        monetary_bins: int = 5
    ):
        """
        Initialize RFM Feature Engineer.

        Args:
            recency_bins: Number of bins for recency scoring
            frequency_bins: Number of bins for frequency scoring
            monetary_bins: Number of bins for monetary scoring
        """
        self.recency_bins = recency_bins
        self.frequency_bins = frequency_bins
        self.monetary_bins = monetary_bins

        logger.info("RFMFeatureEngineer initialized")

    def calculate_rfm(
        self,
        df: pd.DataFrame,
        customer_id: str,
        date_column: str,
        amount_column: str,
        reference_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.

        Args:
            df: Transaction DataFrame
            customer_id: Column name for customer ID
            date_column: Column name for transaction date
            amount_column: Column name for transaction amount
            reference_date: Reference date for recency (default: max date)

        Returns:
            DataFrame with RFM metrics per customer

        Example:
            >>> rfm = engineer.calculate_rfm(
            ...     transactions, 'customer_id', 'date', 'amount'
            ... )
        """
        df = df.copy()

        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Set reference date
        if reference_date is None:
            reference_date = df[date_column].max() + pd.Timedelta(days=1)

        # Calculate RFM metrics
        rfm = df.groupby(customer_id).agg({
            date_column: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id: 'count',  # Frequency
            amount_column: 'sum'  # Monetary
        })

        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm.reset_index()

        logger.info(f"Calculated RFM for {len(rfm)} customers")
        return rfm

    def calculate_rfm_scores(
        self,
        rfm: pd.DataFrame,
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate RFM scores (1-5) for each customer.

        Args:
            rfm: DataFrame with RFM metrics
            labels: Custom labels for bins

        Returns:
            DataFrame with RFM scores

        Example:
            >>> scored = engineer.calculate_rfm_scores(rfm)
        """
        rfm = rfm.copy()

        # Recency: lower is better (inverse scoring)
        rfm['r_score'] = pd.qcut(
            rfm['recency'],
            q=self.recency_bins,
            labels=range(self.recency_bins, 0, -1),
            duplicates='drop'
        ).astype(int)

        # Frequency: higher is better
        rfm['f_score'] = pd.qcut(
            rfm['frequency'].rank(method='first'),
            q=self.frequency_bins,
            labels=range(1, self.frequency_bins + 1),
            duplicates='drop'
        ).astype(int)

        # Monetary: higher is better
        rfm['m_score'] = pd.qcut(
            rfm['monetary'].rank(method='first'),
            q=self.monetary_bins,
            labels=range(1, self.monetary_bins + 1),
            duplicates='drop'
        ).astype(int)

        # Combined RFM score
        rfm['rfm_score'] = (
            rfm['r_score'].astype(str) +
            rfm['f_score'].astype(str) +
            rfm['m_score'].astype(str)
        )

        # Numeric combined score
        rfm['rfm_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

        logger.info("Calculated RFM scores")
        return rfm

    def segment_customers(
        self,
        rfm: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign customers to predefined RFM segments.

        Args:
            rfm: DataFrame with RFM scores

        Returns:
            DataFrame with segment assignments

        Example:
            >>> segmented = engineer.segment_customers(scored_rfm)
        """
        rfm = rfm.copy()

        # Define segment mapping based on RFM scores
        def assign_segment(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']

            # Champions: Best customers
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            # Loyal Customers
            elif f >= 4:
                return 'Loyal Customers'
            # Potential Loyalists
            elif r >= 4 and f >= 2:
                return 'Potential Loyalists'
            # Recent Customers
            elif r >= 4:
                return 'Recent Customers'
            # Promising
            elif r >= 3 and f >= 1 and m >= 2:
                return 'Promising'
            # Customers Needing Attention
            elif r >= 2 and f >= 2:
                return 'Need Attention'
            # About to Sleep
            elif r >= 2:
                return 'About to Sleep'
            # At Risk
            elif r <= 2 and f >= 3:
                return 'At Risk'
            # Cannot Lose Them
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cannot Lose'
            # Hibernating
            elif r <= 2 and f <= 2:
                return 'Hibernating'
            # Lost
            else:
                return 'Lost'

        rfm['segment'] = rfm.apply(assign_segment, axis=1)

        # Log segment distribution
        segment_counts = rfm['segment'].value_counts()
        logger.info(f"Segment distribution:\n{segment_counts}")

        return rfm

    def engineer_features(
        self,
        rfm: pd.DataFrame,
        transactions: Optional[pd.DataFrame] = None,
        customer_id: str = 'customer_id',
        date_column: str = 'date',
        amount_column: str = 'amount'
    ) -> pd.DataFrame:
        """
        Engineer additional behavioral features.

        Args:
            rfm: DataFrame with RFM metrics
            transactions: Original transaction data for additional features
            customer_id: Customer ID column
            date_column: Date column
            amount_column: Amount column

        Returns:
            DataFrame with engineered features

        Example:
            >>> features = engineer.engineer_features(rfm, transactions)
        """
        features = rfm.copy()

        # Basic derived features
        features['avg_order_value'] = features['monetary'] / features['frequency']
        features['monetary_per_day'] = features['monetary'] / (features['recency'] + 1)

        # Log transformations for skewed distributions
        features['log_monetary'] = np.log1p(features['monetary'])
        features['log_frequency'] = np.log1p(features['frequency'])

        # Recency inverse (for positive correlation with value)
        features['recency_inverse'] = 1 / (features['recency'] + 1)

        if transactions is not None:
            # Additional features from transactions
            trans = transactions.copy()
            trans[date_column] = pd.to_datetime(trans[date_column])

            # Transaction variability
            trans_stats = trans.groupby(customer_id)[amount_column].agg([
                'std', 'min', 'max', 'median'
            ]).reset_index()
            trans_stats.columns = [customer_id, 'amount_std', 'amount_min',
                                   'amount_max', 'amount_median']

            features = features.merge(trans_stats, on=customer_id, how='left')

            # Fill NaN for customers with single transaction
            features['amount_std'] = features['amount_std'].fillna(0)

            # Coefficient of variation
            features['amount_cv'] = features['amount_std'] / (features['avg_order_value'] + 1)

            # Purchase interval statistics
            interval_stats = self._calculate_purchase_intervals(
                trans, customer_id, date_column
            )
            features = features.merge(interval_stats, on=customer_id, how='left')

            # Time-based features
            time_features = self._calculate_time_features(
                trans, customer_id, date_column, amount_column
            )
            features = features.merge(time_features, on=customer_id, how='left')

        logger.info(f"Engineered {len(features.columns)} features")
        return features

    def _calculate_purchase_intervals(
        self,
        df: pd.DataFrame,
        customer_id: str,
        date_column: str
    ) -> pd.DataFrame:
        """Calculate statistics on purchase intervals."""
        intervals = []

        for cust_id in df[customer_id].unique():
            cust_trans = df[df[customer_id] == cust_id].sort_values(date_column)

            if len(cust_trans) > 1:
                days_between = cust_trans[date_column].diff().dt.days.dropna()
                intervals.append({
                    customer_id: cust_id,
                    'avg_days_between': days_between.mean(),
                    'std_days_between': days_between.std(),
                    'min_days_between': days_between.min(),
                    'max_days_between': days_between.max()
                })
            else:
                intervals.append({
                    customer_id: cust_id,
                    'avg_days_between': np.nan,
                    'std_days_between': np.nan,
                    'min_days_between': np.nan,
                    'max_days_between': np.nan
                })

        return pd.DataFrame(intervals)

    def _calculate_time_features(
        self,
        df: pd.DataFrame,
        customer_id: str,
        date_column: str,
        amount_column: str
    ) -> pd.DataFrame:
        """Calculate time-based behavioral features."""
        df = df.copy()
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df[date_column].dt.month

        # Aggregate by customer
        time_features = df.groupby(customer_id).agg({
            'is_weekend': 'mean',
            'hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).reset_index()

        time_features.columns = [
            customer_id, 'weekend_purchase_ratio',
            'preferred_hour', 'preferred_day'
        ]

        # Calculate trend (growth in spending)
        trend = df.groupby(customer_id).apply(
            lambda x: self._calculate_trend(x, date_column, amount_column)
        ).reset_index()
        trend.columns = [customer_id, 'spending_trend']

        time_features = time_features.merge(trend, on=customer_id, how='left')

        return time_features

    def _calculate_trend(
        self,
        df: pd.DataFrame,
        date_column: str,
        amount_column: str
    ) -> float:
        """Calculate spending trend using linear regression slope."""
        if len(df) < 2:
            return 0

        df = df.sort_values(date_column)
        x = np.arange(len(df))
        y = df[amount_column].values

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def get_feature_summary(
        self,
        features: pd.DataFrame,
        numeric_only: bool = True
    ) -> pd.DataFrame:
        """
        Get summary statistics for engineered features.

        Args:
            features: DataFrame with engineered features
            numeric_only: Only include numeric columns

        Returns:
            Summary statistics DataFrame
        """
        if numeric_only:
            summary = features.select_dtypes(include=[np.number]).describe()
        else:
            summary = features.describe(include='all')

        return summary.T

    def identify_outlier_customers(
        self,
        features: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Identify outlier customers based on feature values.

        Args:
            features: DataFrame with features
            columns: Columns to check for outliers
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outlier customers
        """
        outlier_mask = pd.Series([False] * len(features))

        for col in columns:
            if col not in features.columns:
                continue

            data = features[col]

            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask |= (data < Q1 - threshold * IQR) | (data > Q3 + threshold * IQR)
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data.dropna()))
                outlier_mask |= z_scores > threshold

        outliers = features[outlier_mask].copy()
        logger.info(f"Identified {len(outliers)} outlier customers")

        return outliers

    def get_segment_profiles(
        self,
        rfm: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate profiles for each customer segment.

        Args:
            rfm: DataFrame with RFM scores and segments
            features: Additional features to include

        Returns:
            DataFrame with segment profiles
        """
        if 'segment' not in rfm.columns:
            rfm = self.segment_customers(rfm)

        # Basic profile
        profile_cols = ['recency', 'frequency', 'monetary',
                       'r_score', 'f_score', 'm_score', 'rfm_total']

        if features is not None:
            # Merge features
            rfm = rfm.merge(
                features.drop(columns=['recency', 'frequency', 'monetary'], errors='ignore'),
                on=rfm.columns[0],
                how='left'
            )
            profile_cols.extend([
                'avg_order_value', 'monetary_per_day'
            ])

        profile = rfm.groupby('segment')[profile_cols].agg(['mean', 'median', 'count'])

        # Flatten column names
        profile.columns = ['_'.join(col).strip() for col in profile.columns.values]

        # Add percentage of total
        total_customers = len(rfm)
        profile['customer_count'] = rfm.groupby('segment').size()
        profile['customer_percentage'] = profile['customer_count'] / total_customers * 100

        return profile.sort_values('monetary_mean', ascending=False)
