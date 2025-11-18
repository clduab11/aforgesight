"""
Data Loading and Validation Module
===================================

Provides enterprise-grade data loading capabilities with validation,
type inference, and error handling for retail analytics datasets.

Usage:
    from src.common import DataLoader

    loader = DataLoader(config_path="config/settings.yaml")
    df = loader.load_csv("data/sales.csv", date_columns=["date"])

    # Validate data
    is_valid, report = loader.validate_data(df, schema="sales")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from datetime import datetime
import yaml
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Enterprise data loader with validation and preprocessing capabilities.

    Attributes:
        config (dict): Configuration dictionary loaded from YAML
        supported_formats (list): List of supported file formats

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_csv("sales_data.csv", date_columns=["transaction_date"])
        >>> print(f"Loaded {len(df)} records")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader with optional configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.supported_formats = ['.csv', '.parquet', '.json', '.xlsx', '.feather']
        logger.info("DataLoader initialized")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        default_config = {
            'data': {
                'date_formats': ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
                'missing_value_threshold': 0.3,
                'min_data_points': 30
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return default_config

    def load_csv(
        self,
        filepath: Union[str, Path],
        date_columns: Optional[List[str]] = None,
        parse_dates: bool = True,
        dtype: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with automatic date parsing and type inference.

        Args:
            filepath: Path to CSV file
            date_columns: List of column names to parse as dates
            parse_dates: Whether to automatically detect and parse dates
            dtype: Dictionary of column dtypes
            chunk_size: If specified, returns iterator for large files
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            DataFrame with loaded and parsed data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported

        Example:
            >>> df = loader.load_csv("sales.csv", date_columns=["date", "created_at"])
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        logger.info(f"Loading data from {filepath}")

        # Prepare read arguments
        read_kwargs = {
            'dtype': dtype,
            'low_memory': False,
            **kwargs
        }

        # Handle date columns
        if date_columns:
            read_kwargs['parse_dates'] = date_columns

        # Load data
        if chunk_size:
            return pd.read_csv(filepath, chunksize=chunk_size, **read_kwargs)

        df = pd.read_csv(filepath, **read_kwargs)

        # Auto-detect and parse remaining date columns
        if parse_dates and not date_columns:
            df = self._auto_parse_dates(df)

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def load_parquet(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load Parquet file for optimized large dataset handling.

        Args:
            filepath: Path to Parquet file
            **kwargs: Additional arguments passed to pd.read_parquet

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading parquet from {filepath}")
        df = pd.read_parquet(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} records")
        return df

    def load_excel(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Excel file with sheet selection.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index to load
            **kwargs: Additional arguments passed to pd.read_excel

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading Excel from {filepath}, sheet: {sheet_name}")
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        logger.info(f"Loaded {len(df)} records")
        return df

    def _auto_parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and parse date columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with parsed date columns
        """
        date_formats = self.config.get('data', {}).get('date_formats', [])

        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column looks like dates
                sample = df[col].dropna().head(100)
                if len(sample) == 0:
                    continue

                for fmt in date_formats:
                    try:
                        parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                        if parsed.notna().sum() > len(sample) * 0.8:
                            df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                            logger.debug(f"Auto-parsed date column: {col}")
                            break
                    except Exception:
                        continue

        return df

    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        schema: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate DataFrame against requirements and generate quality report.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            schema: Schema type ('sales', 'customers', 'transactions')

        Returns:
            Tuple of (is_valid, validation_report)

        Example:
            >>> is_valid, report = loader.validate_data(df, required_columns=["date", "amount"])
            >>> if not is_valid:
            ...     print(report['errors'])
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        # Check minimum data points
        min_points = self.config.get('data', {}).get('min_data_points', 30)
        if len(df) < min_points:
            report['errors'].append(f"Insufficient data: {len(df)} < {min_points} required")
            report['is_valid'] = False

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                report['errors'].append(f"Missing required columns: {missing}")
                report['is_valid'] = False

        # Check missing values
        missing_threshold = self.config.get('data', {}).get('missing_value_threshold', 0.3)
        for col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > missing_threshold:
                report['warnings'].append(
                    f"High missing ratio in '{col}': {missing_ratio:.2%}"
                )

        # Generate statistics
        report['statistics'] = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isna().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }

        # Schema-specific validation
        if schema:
            schema_validation = self._validate_schema(df, schema)
            report['errors'].extend(schema_validation.get('errors', []))
            report['warnings'].extend(schema_validation.get('warnings', []))
            if schema_validation.get('errors'):
                report['is_valid'] = False

        return report['is_valid'], report

    def _validate_schema(self, df: pd.DataFrame, schema: str) -> Dict[str, List[str]]:
        """Validate DataFrame against predefined schemas."""
        schemas = {
            'sales': {
                'required': ['date', 'sales'],
                'numeric': ['sales'],
                'datetime': ['date']
            },
            'customers': {
                'required': ['customer_id'],
                'numeric': [],
                'datetime': []
            },
            'transactions': {
                'required': ['transaction_id', 'amount', 'timestamp'],
                'numeric': ['amount'],
                'datetime': ['timestamp']
            }
        }

        result = {'errors': [], 'warnings': []}

        if schema not in schemas:
            result['warnings'].append(f"Unknown schema: {schema}")
            return result

        schema_def = schemas[schema]

        # Check required columns
        missing = set(schema_def['required']) - set(df.columns)
        if missing:
            result['errors'].append(f"Schema '{schema}' missing columns: {missing}")

        # Check numeric columns
        for col in schema_def['numeric']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                result['warnings'].append(f"Column '{col}' should be numeric")

        # Check datetime columns
        for col in schema_def['datetime']:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                result['warnings'].append(f"Column '{col}' should be datetime")

        return result

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()

        # Categorical columns summary
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }

        return summary

    def merge_datasets(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: Union[str, List[str]],
        how: str = 'inner',
        validate: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Merge two datasets with validation.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Column(s) to merge on
            how: Type of merge ('inner', 'left', 'right', 'outer')
            validate: Merge validation ('one_to_one', 'one_to_many', etc.)

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging datasets on {on} using {how} join")

        result = pd.merge(left, right, on=on, how=how, validate=validate)

        logger.info(f"Merge result: {len(result)} records")
        return result
