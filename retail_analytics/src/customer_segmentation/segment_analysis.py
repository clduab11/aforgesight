"""
Segment Analysis Module
=======================

Advanced analysis and insights generation for customer segments.

Usage:
    from src.customer_segmentation import SegmentAnalyzer

    analyzer = SegmentAnalyzer()
    insights = analyzer.analyze_segments(df, 'cluster')
    migrations = analyzer.analyze_segment_migration(df_t1, df_t2)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from scipy import stats
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class SegmentAnalyzer:
    """
    Advanced analysis toolkit for customer segments.

    Provides statistical analysis, segment comparisons, and
    actionable insights generation.

    Example:
        >>> analyzer = SegmentAnalyzer()
        >>> insights = analyzer.analyze_segments(df, 'cluster')
        >>> print(insights['summary'])
    """

    def __init__(self):
        """Initialize SegmentAnalyzer."""
        logger.info("SegmentAnalyzer initialized")

    def analyze_segments(
        self,
        df: pd.DataFrame,
        segment_column: str,
        value_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive segment analysis.

        Args:
            df: DataFrame with segment assignments
            segment_column: Column containing segment labels
            value_columns: Columns to analyze (None for all numeric)

        Returns:
            Dictionary with analysis results

        Example:
            >>> results = analyzer.analyze_segments(df, 'cluster')
        """
        if value_columns is None:
            value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            value_columns = [c for c in value_columns if c != segment_column]

        results = {
            'segment_sizes': self._analyze_segment_sizes(df, segment_column),
            'segment_statistics': self._calculate_segment_statistics(
                df, segment_column, value_columns
            ),
            'statistical_tests': self._perform_statistical_tests(
                df, segment_column, value_columns
            ),
            'feature_importance': self._calculate_feature_importance(
                df, segment_column, value_columns
            ),
            'summary': self._generate_summary(df, segment_column, value_columns)
        }

        return results

    def _analyze_segment_sizes(
        self,
        df: pd.DataFrame,
        segment_column: str
    ) -> pd.DataFrame:
        """Analyze segment size distribution."""
        sizes = df[segment_column].value_counts().reset_index()
        sizes.columns = ['segment', 'count']
        sizes['percentage'] = sizes['count'] / len(df) * 100
        sizes['cumulative_pct'] = sizes['percentage'].cumsum()

        return sizes

    def _calculate_segment_statistics(
        self,
        df: pd.DataFrame,
        segment_column: str,
        value_columns: List[str]
    ) -> pd.DataFrame:
        """Calculate detailed statistics for each segment."""
        stats_list = []

        for segment in df[segment_column].unique():
            segment_data = df[df[segment_column] == segment]

            for col in value_columns:
                if col not in segment_data.columns:
                    continue

                values = segment_data[col].dropna()

                stats_dict = {
                    'segment': segment,
                    'feature': col,
                    'count': len(values),
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'skewness': values.skew(),
                    'kurtosis': values.kurtosis()
                }

                stats_list.append(stats_dict)

        return pd.DataFrame(stats_list)

    def _perform_statistical_tests(
        self,
        df: pd.DataFrame,
        segment_column: str,
        value_columns: List[str]
    ) -> Dict[str, Any]:
        """Perform statistical tests between segments."""
        test_results = {}

        for col in value_columns:
            if col not in df.columns:
                continue

            # Group data by segment
            groups = [
                df[df[segment_column] == seg][col].dropna().values
                for seg in df[segment_column].unique()
            ]

            # Filter out empty groups
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                continue

            # Kruskal-Wallis H-test (non-parametric ANOVA)
            try:
                h_stat, h_pvalue = stats.kruskal(*groups)
                test_results[col] = {
                    'kruskal_h_statistic': h_stat,
                    'kruskal_p_value': h_pvalue,
                    'significant': h_pvalue < 0.05
                }
            except Exception as e:
                logger.warning(f"Statistical test failed for {col}: {e}")

        return test_results

    def _calculate_feature_importance(
        self,
        df: pd.DataFrame,
        segment_column: str,
        value_columns: List[str]
    ) -> pd.DataFrame:
        """Calculate feature importance for segment differentiation."""
        importance_scores = []

        overall_means = df[value_columns].mean()
        overall_stds = df[value_columns].std()

        for col in value_columns:
            if col not in df.columns:
                continue

            # Calculate between-segment variance
            segment_means = df.groupby(segment_column)[col].mean()
            between_variance = segment_means.var()

            # Calculate within-segment variance
            within_variance = df.groupby(segment_column)[col].var().mean()

            # F-ratio as importance measure
            if within_variance > 0:
                f_ratio = between_variance / within_variance
            else:
                f_ratio = 0

            # Coefficient of variation across segments
            cv = segment_means.std() / (segment_means.mean() + 1e-10)

            importance_scores.append({
                'feature': col,
                'f_ratio': f_ratio,
                'coefficient_of_variation': cv,
                'between_variance': between_variance,
                'within_variance': within_variance
            })

        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('f_ratio', ascending=False)

        return importance_df

    def _generate_summary(
        self,
        df: pd.DataFrame,
        segment_column: str,
        value_columns: List[str]
    ) -> str:
        """Generate text summary of segment analysis."""
        n_segments = df[segment_column].nunique()
        n_customers = len(df)

        summary_parts = [
            f"Segment Analysis Summary",
            f"=" * 40,
            f"Total customers: {n_customers:,}",
            f"Number of segments: {n_segments}",
            f"Features analyzed: {len(value_columns)}",
            ""
        ]

        # Segment sizes
        sizes = df[segment_column].value_counts()
        summary_parts.append("Segment Distribution:")
        for seg, count in sizes.items():
            pct = count / n_customers * 100
            summary_parts.append(f"  Segment {seg}: {count:,} ({pct:.1f}%)")

        summary_parts.append("")

        # Key differentiators
        if value_columns:
            summary_parts.append("Key Segment Differentiators:")
            for col in value_columns[:3]:  # Top 3
                if col in df.columns:
                    segment_means = df.groupby(segment_column)[col].mean()
                    best_seg = segment_means.idxmax()
                    worst_seg = segment_means.idxmin()
                    summary_parts.append(
                        f"  {col}: Highest in Segment {best_seg}, "
                        f"Lowest in Segment {worst_seg}"
                    )

        return "\n".join(summary_parts)

    def analyze_segment_migration(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        customer_id: str,
        segment_column: str
    ) -> pd.DataFrame:
        """
        Analyze customer migration between segments over time.

        Args:
            df_before: DataFrame with earlier segment assignments
            df_after: DataFrame with later segment assignments
            customer_id: Customer ID column
            segment_column: Segment column name

        Returns:
            Migration matrix DataFrame
        """
        # Merge datasets
        merged = df_before[[customer_id, segment_column]].merge(
            df_after[[customer_id, segment_column]],
            on=customer_id,
            suffixes=('_before', '_after')
        )

        # Create migration matrix
        migration = pd.crosstab(
            merged[f'{segment_column}_before'],
            merged[f'{segment_column}_after'],
            margins=True
        )

        # Calculate percentages
        migration_pct = migration.div(migration.iloc[:-1, -1], axis=0) * 100

        logger.info(f"Analyzed migration for {len(merged)} customers")
        return migration

    def calculate_segment_value(
        self,
        df: pd.DataFrame,
        segment_column: str,
        monetary_column: str,
        frequency_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate customer lifetime value proxy by segment.

        Args:
            df: DataFrame with customer data
            segment_column: Segment column
            monetary_column: Monetary value column
            frequency_column: Purchase frequency column

        Returns:
            DataFrame with segment value metrics
        """
        value_metrics = df.groupby(segment_column).agg({
            monetary_column: ['sum', 'mean', 'median'],
            segment_column: 'count'
        })

        value_metrics.columns = [
            'total_revenue', 'avg_revenue', 'median_revenue', 'customer_count'
        ]

        # Calculate segment contribution
        total_revenue = value_metrics['total_revenue'].sum()
        value_metrics['revenue_share'] = value_metrics['total_revenue'] / total_revenue * 100

        # Revenue per customer
        value_metrics['revenue_per_customer'] = (
            value_metrics['total_revenue'] / value_metrics['customer_count']
        )

        if frequency_column and frequency_column in df.columns:
            freq_stats = df.groupby(segment_column)[frequency_column].mean()
            value_metrics['avg_frequency'] = freq_stats

            # Simple CLV proxy
            value_metrics['clv_proxy'] = (
                value_metrics['avg_revenue'] * value_metrics['avg_frequency']
            )

        return value_metrics.sort_values('total_revenue', ascending=False)

    def find_segment_drivers(
        self,
        df: pd.DataFrame,
        segment_column: str,
        feature_columns: List[str],
        target_segment: Any
    ) -> pd.DataFrame:
        """
        Find key drivers that distinguish a target segment.

        Args:
            df: DataFrame with features
            segment_column: Segment column
            feature_columns: Features to analyze
            target_segment: Segment to analyze

        Returns:
            DataFrame with driver analysis
        """
        target_data = df[df[segment_column] == target_segment]
        other_data = df[df[segment_column] != target_segment]

        drivers = []

        for col in feature_columns:
            if col not in df.columns:
                continue

            target_mean = target_data[col].mean()
            other_mean = other_data[col].mean()

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (target_data[col].std()**2 + other_data[col].std()**2) / 2
            )
            if pooled_std > 0:
                cohens_d = (target_mean - other_mean) / pooled_std
            else:
                cohens_d = 0

            # Statistical test
            try:
                t_stat, p_value = stats.mannwhitneyu(
                    target_data[col].dropna(),
                    other_data[col].dropna(),
                    alternative='two-sided'
                )
            except Exception:
                t_stat, p_value = np.nan, np.nan

            drivers.append({
                'feature': col,
                'target_mean': target_mean,
                'other_mean': other_mean,
                'difference': target_mean - other_mean,
                'difference_pct': ((target_mean - other_mean) / (other_mean + 1e-10)) * 100,
                'effect_size': cohens_d,
                'p_value': p_value
            })

        drivers_df = pd.DataFrame(drivers)
        drivers_df = drivers_df.sort_values('effect_size', key=abs, ascending=False)

        return drivers_df

    def generate_segment_personas(
        self,
        df: pd.DataFrame,
        segment_column: str,
        profiles: Dict[int, Dict[str, Any]]
    ) -> Dict[int, str]:
        """
        Generate descriptive personas for each segment.

        Args:
            df: DataFrame with customer data
            segment_column: Segment column
            profiles: Cluster profiles

        Returns:
            Dictionary mapping segment to persona description
        """
        personas = {}

        for segment, profile in profiles.items():
            # Extract key metrics
            size = profile.get('size', 0)
            pct = profile.get('percentage', 0)

            recency = profile.get('avg_recency', 0)
            frequency = profile.get('avg_frequency', 0)
            monetary = profile.get('avg_monetary', 0)

            # Determine characteristics
            characteristics = []

            # Recency
            median_recency = np.median([p.get('avg_recency', 0) for p in profiles.values()])
            if recency < median_recency * 0.5:
                characteristics.append("very active")
            elif recency < median_recency:
                characteristics.append("recently engaged")
            elif recency > median_recency * 1.5:
                characteristics.append("inactive")
            else:
                characteristics.append("moderately active")

            # Frequency
            median_freq = np.median([p.get('avg_frequency', 0) for p in profiles.values()])
            if frequency > median_freq * 1.5:
                characteristics.append("frequent buyer")
            elif frequency < median_freq * 0.5:
                characteristics.append("occasional buyer")

            # Monetary
            median_monetary = np.median([p.get('avg_monetary', 0) for p in profiles.values()])
            if monetary > median_monetary * 2:
                characteristics.append("high spender")
            elif monetary > median_monetary:
                characteristics.append("above-average spender")
            elif monetary < median_monetary * 0.5:
                characteristics.append("budget-conscious")

            # Build persona
            persona = f"Segment {segment} ({size:,} customers, {pct:.1f}%): "
            persona += ", ".join(characteristics) + ". "
            persona += f"Average spend: ${monetary:.2f}, "
            persona += f"Purchase frequency: {frequency:.1f}, "
            persona += f"Days since last purchase: {recency:.0f}"

            personas[segment] = persona

        return personas

    def calculate_segment_overlap(
        self,
        df: pd.DataFrame,
        segment_column: str,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Calculate overlap between segments based on feature distributions.

        Args:
            df: DataFrame with features
            segment_column: Segment column
            feature_columns: Features to analyze

        Returns:
            Overlap matrix DataFrame
        """
        segments = df[segment_column].unique()
        n_segments = len(segments)

        overlap_matrix = np.zeros((n_segments, n_segments))

        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    # Calculate overlap based on feature distributions
                    overlaps = []
                    for col in feature_columns:
                        data1 = df[df[segment_column] == seg1][col].dropna()
                        data2 = df[df[segment_column] == seg2][col].dropna()

                        if len(data1) > 0 and len(data2) > 0:
                            # Use Bhattacharyya coefficient
                            overlap = self._bhattacharyya_coefficient(data1, data2)
                            overlaps.append(overlap)

                    if overlaps:
                        overlap_matrix[i, j] = np.mean(overlaps)

        return pd.DataFrame(
            overlap_matrix,
            index=[f'Segment_{s}' for s in segments],
            columns=[f'Segment_{s}' for s in segments]
        )

    def _bhattacharyya_coefficient(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """Calculate Bhattacharyya coefficient between two distributions."""
        # Create common bin edges
        combined = np.concatenate([data1, data2])
        bins = np.linspace(combined.min(), combined.max(), n_bins)

        # Calculate histograms
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)

        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))

        return bc
