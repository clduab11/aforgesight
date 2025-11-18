"""
Data Preprocessing Module
=========================

Comprehensive data preprocessing for retail analytics including
cleaning, imputation, scaling, and feature transformation.

Usage:
    from src.common import Preprocessor

    preprocessor = Preprocessor()
    df_clean = preprocessor.clean_data(df)
    df_scaled = preprocessor.scale_features(df_clean, columns=['amount', 'quantity'])
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class Preprocessor:
    """
    Enterprise-grade data preprocessor for retail analytics.

    Provides methods for:
    - Data cleaning and validation
    - Missing value imputation
    - Outlier detection and treatment
    - Feature scaling and transformation
    - Time series preprocessing

    Example:
        >>> preprocessor = Preprocessor()
        >>> df_clean = preprocessor.clean_data(df)
        >>> df_scaled = preprocessor.scale_features(df_clean, ['price', 'quantity'])
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize Preprocessor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        logger.info("Preprocessor initialized")

    def clean_data(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        remove_empty_cols: bool = True,
        standardize_columns: bool = True
    ) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.

        Args:
            df: Input DataFrame
            remove_duplicates: Remove duplicate rows
            handle_missing: Handle missing values
            remove_empty_cols: Remove columns with all missing values
            standardize_columns: Standardize column names

        Returns:
            Cleaned DataFrame

        Example:
            >>> df_clean = preprocessor.clean_data(df)
        """
        df = df.copy()
        original_shape = df.shape

        logger.info(f"Starting data cleaning. Shape: {original_shape}")

        # Standardize column names
        if standardize_columns:
            df.columns = (
                df.columns
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('[^a-z0-9_]', '', regex=True)
            )

        # Remove empty columns
        if remove_empty_cols:
            empty_cols = df.columns[df.isna().all()].tolist()
            if empty_cols:
                df = df.drop(columns=empty_cols)
                logger.info(f"Removed {len(empty_cols)} empty columns")

        # Remove duplicates
        if remove_duplicates:
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {n_duplicates} duplicate rows")

        # Handle missing values (basic)
        if handle_missing:
            # For numeric columns with < 5% missing, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                missing_ratio = df[col].isna().sum() / len(df)
                if 0 < missing_ratio < 0.05:
                    df[col] = df[col].fillna(df[col].median())

        logger.info(f"Cleaning complete. Shape: {original_shape} -> {df.shape}")
        return df

    def impute_missing(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        columns: Optional[List[str]] = None,
        use_knn: bool = False,
        n_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Impute missing values with various strategies.

        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'most_frequent')
            categorical_strategy: Strategy for categorical columns
            columns: Specific columns to impute (None for all)
            use_knn: Use KNN imputation for numeric columns
            n_neighbors: Number of neighbors for KNN

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        if columns:
            target_cols = columns
        else:
            target_cols = df.columns.tolist()

        # Separate numeric and categorical columns
        numeric_cols = df[target_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df[target_cols].select_dtypes(include=['object', 'category']).columns.tolist()

        # Impute numeric columns
        if numeric_cols:
            if use_knn:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric_knn'] = imputer
            else:
                imputer = SimpleImputer(strategy=numeric_strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = imputer
            logger.info(f"Imputed {len(numeric_cols)} numeric columns")

        # Impute categorical columns
        if cat_cols:
            imputer = SimpleImputer(strategy=categorical_strategy)
            df[cat_cols] = imputer.fit_transform(df[cat_cols])
            self.imputers['categorical'] = imputer
            logger.info(f"Imputed {len(cat_cols)} categorical columns")

        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore',
        threshold: float = 3.0,
        treatment: str = 'clip'
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.

        Args:
            df: Input DataFrame
            columns: Columns to process (None for all numeric)
            method: Detection method ('zscore', 'iqr', 'percentile')
            threshold: Threshold for outlier detection
            treatment: Treatment method ('clip', 'remove', 'nan')

        Returns:
            DataFrame with treated outliers

        Example:
            >>> df = preprocessor.handle_outliers(df, method='iqr', treatment='clip')
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_counts = {}

        for col in columns:
            if col not in df.columns:
                continue

            data = df[col].dropna()

            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > threshold
            elif method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask = (data < lower) | (data > upper)
            elif method == 'percentile':
                lower = data.quantile(0.01)
                upper = data.quantile(0.99)
                outlier_mask = (data < lower) | (data > upper)
            else:
                raise ValueError(f"Unknown method: {method}")

            n_outliers = outlier_mask.sum()
            outlier_counts[col] = n_outliers

            if n_outliers > 0:
                if treatment == 'clip':
                    if method == 'iqr':
                        df[col] = df[col].clip(lower=lower, upper=upper)
                    else:
                        lower = data[~outlier_mask].min()
                        upper = data[~outlier_mask].max()
                        df[col] = df[col].clip(lower=lower, upper=upper)
                elif treatment == 'remove':
                    df = df[~df.index.isin(data[outlier_mask].index)]
                elif treatment == 'nan':
                    df.loc[data[outlier_mask].index, col] = np.nan

        total_outliers = sum(outlier_counts.values())
        logger.info(f"Handled {total_outliers} outliers across {len(columns)} columns")

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numeric features using various methods.

        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (False for transform only)

        Returns:
            DataFrame with scaled features

        Example:
            >>> df = preprocessor.scale_features(df, ['price', 'quantity'], method='standard')
        """
        df = df.copy()

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if fit:
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers[method] = scaler
        else:
            if method not in self.scalers:
                raise ValueError(f"Scaler '{method}' not fitted. Call with fit=True first.")
            df[columns] = self.scalers[method].transform(df[columns])

        logger.info(f"Scaled {len(columns)} columns using {method} method")
        return df

    def inverse_scale(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Inverse transform scaled features.

        Args:
            df: DataFrame with scaled features
            columns: Columns to inverse transform
            method: Scaling method used

        Returns:
            DataFrame with original scale
        """
        df = df.copy()

        if method not in self.scalers:
            raise ValueError(f"Scaler '{method}' not found")

        df[columns] = self.scalers[method].inverse_transform(df[columns])
        return df

    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        freq: Optional[str] = None,
        fill_method: str = 'ffill',
        aggregate: str = 'sum'
    ) -> pd.DataFrame:
        """
        Prepare data for time series analysis.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column to analyze
            freq: Frequency for resampling (None for auto-detect)
            fill_method: Method to fill missing dates ('ffill', 'bfill', 'interpolate')
            aggregate: Aggregation function for duplicate dates

        Returns:
            Time series DataFrame with continuous dates

        Example:
            >>> ts_df = preprocessor.prepare_time_series(
            ...     df, 'date', 'sales', freq='D'
            ... )
        """
        df = df.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Set date as index
        df = df.set_index(date_column)

        # Aggregate duplicates
        if aggregate == 'sum':
            df = df.groupby(df.index)[value_column].sum().to_frame()
        elif aggregate == 'mean':
            df = df.groupby(df.index)[value_column].mean().to_frame()
        else:
            df = df.groupby(df.index)[value_column].agg(aggregate).to_frame()

        # Auto-detect frequency if not provided
        if freq is None:
            freq = pd.infer_freq(df.index)
            if freq is None:
                # Default to daily
                freq = 'D'
                logger.warning("Could not infer frequency, using 'D' (daily)")

        # Resample to ensure continuous dates
        df = df.resample(freq).sum()

        # Fill missing values
        if fill_method == 'interpolate':
            df = df.interpolate(method='time')
        else:
            df = df.fillna(method=fill_method)

        # Fill any remaining NaN with 0
        df = df.fillna(0)

        logger.info(f"Prepared time series with {len(df)} observations at {freq} frequency")
        return df.reset_index()

    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lags: List[int],
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series data.

        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag periods
            date_column: Date column (for sorting)

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        if date_column:
            df = df.sort_values(date_column)

        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)

        logger.info(f"Created {len(lags)} lag features for {column}")
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max'],
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            df: Input DataFrame
            column: Column to calculate rolling features for
            windows: List of window sizes
            functions: Aggregation functions to apply
            date_column: Date column (for sorting)

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        if date_column:
            df = df.sort_values(date_column)

        for window in windows:
            rolling = df[column].rolling(window=window, min_periods=1)
            for func in functions:
                df[f'{column}_rolling_{window}_{func}'] = getattr(rolling, func)()

        logger.info(f"Created rolling features for windows {windows}")
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot',
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            columns: Columns to encode
            method: Encoding method ('onehot', 'label', 'target')
            drop_first: Drop first category for one-hot encoding

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        logger.info(f"Encoded {len(columns)} categorical columns using {method}")
        return df

    def detect_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.

        Args:
            reference: Reference/baseline DataFrame
            current: Current DataFrame to compare
            columns: Columns to check (None for all numeric)
            threshold: P-value threshold for significance

        Returns:
            Dictionary with drift detection results
        """
        if columns is None:
            columns = reference.select_dtypes(include=[np.number]).columns.tolist()

        results = {'drifted_columns': [], 'details': {}}

        for col in columns:
            if col not in current.columns:
                continue

            # Kolmogorov-Smirnov test
            stat, p_value = stats.ks_2samp(
                reference[col].dropna(),
                current[col].dropna()
            )

            is_drifted = p_value < threshold

            results['details'][col] = {
                'ks_statistic': stat,
                'p_value': p_value,
                'is_drifted': is_drifted
            }

            if is_drifted:
                results['drifted_columns'].append(col)

        logger.info(f"Drift detection complete. {len(results['drifted_columns'])} columns drifted")
        return results
