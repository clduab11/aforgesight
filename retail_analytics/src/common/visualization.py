"""
Visualization Module
====================

Enterprise visualization capabilities for retail analytics with
support for static and interactive plots.

Usage:
    from src.common import Visualizer

    viz = Visualizer(output_dir="outputs/plots")
    viz.plot_time_series(df, 'date', 'sales', title='Daily Sales')
    viz.plot_clusters(df, 'component_1', 'component_2', 'cluster')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Visualizer:
    """
    Enterprise visualization toolkit for retail analytics.

    Supports both static (matplotlib/seaborn) and interactive (plotly)
    visualizations with consistent styling and export capabilities.

    Example:
        >>> viz = Visualizer(output_dir="outputs/plots")
        >>> viz.plot_time_series(df, 'date', 'sales')
        >>> viz.plot_distribution(df, 'amount', title='Transaction Distribution')
    """

    def __init__(
        self,
        output_dir: str = "outputs/plots",
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        palette: str = "husl"
    ):
        """
        Initialize Visualizer.

        Args:
            output_dir: Directory for saving plots
            style: Matplotlib style
            figsize: Default figure size
            dpi: Resolution for saved figures
            palette: Color palette
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.palette = palette

        # Set style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8-whitegrid')

        sns.set_palette(palette)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")

    def plot_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        title: str = "Time Series",
        forecast_df: Optional[pd.DataFrame] = None,
        confidence_interval: Optional[Tuple[str, str]] = None,
        save_name: Optional[str] = None,
        interactive: bool = False
    ) -> Optional[Any]:
        """
        Plot time series data with optional forecast and confidence intervals.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column
            title: Plot title
            forecast_df: DataFrame with forecasted values
            confidence_interval: Tuple of (lower, upper) column names
            save_name: Filename for saving plot
            interactive: Use plotly for interactive plot

        Returns:
            Figure object if interactive, None otherwise
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_time_series_interactive(
                df, date_column, value_column, title,
                forecast_df, confidence_interval, save_name
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot actual values
        ax.plot(df[date_column], df[value_column], label='Actual', linewidth=2)

        # Plot forecast if provided
        if forecast_df is not None:
            ax.plot(
                forecast_df[date_column],
                forecast_df[value_column],
                label='Forecast',
                linestyle='--',
                linewidth=2
            )

            # Plot confidence interval
            if confidence_interval:
                lower_col, upper_col = confidence_interval
                ax.fill_between(
                    forecast_df[date_column],
                    forecast_df[lower_col],
                    forecast_df[upper_col],
                    alpha=0.3,
                    label='95% CI'
                )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(value_column.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")

        plt.show()
        return None

    def _plot_time_series_interactive(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        title: str,
        forecast_df: Optional[pd.DataFrame],
        confidence_interval: Optional[Tuple[str, str]],
        save_name: Optional[str]
    ) -> go.Figure:
        """Create interactive time series plot with plotly."""
        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[value_column],
            mode='lines',
            name='Actual',
            line=dict(width=2)
        ))

        # Forecast
        if forecast_df is not None:
            fig.add_trace(go.Scatter(
                x=forecast_df[date_column],
                y=forecast_df[value_column],
                mode='lines',
                name='Forecast',
                line=dict(dash='dash', width=2)
            ))

            # Confidence interval
            if confidence_interval:
                lower_col, upper_col = confidence_interval
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df[date_column], forecast_df[date_column][::-1]]),
                    y=pd.concat([forecast_df[upper_col], forecast_df[lower_col][::-1]]),
                    fill='toself',
                    fillcolor='rgba(68, 68, 68, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI'
                ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=value_column.replace('_', ' ').title(),
            hovermode='x unified'
        )

        if save_name:
            save_path = self.output_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved interactive plot: {save_path}")

        return fig

    def plot_forecast_components(
        self,
        components: Dict[str, pd.DataFrame],
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot forecast components (trend, seasonality, etc.).

        Args:
            components: Dictionary with component DataFrames
            save_name: Filename for saving plot
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=(self.figsize[0], 4 * n_components))

        if n_components == 1:
            axes = [axes]

        for ax, (name, data) in zip(axes, components.items()):
            if 'ds' in data.columns:
                ax.plot(data['ds'], data[name])
                ax.set_xlabel('Date')
            else:
                ax.plot(data.index, data.values)

            ax.set_ylabel(name.replace('_', ' ').title())
            ax.set_title(f'{name.replace("_", " ").title()} Component')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved components plot: {save_path}")

        plt.show()

    def plot_clusters(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        cluster_column: str,
        title: str = "Cluster Visualization",
        show_centroids: bool = True,
        centroids: Optional[np.ndarray] = None,
        save_name: Optional[str] = None,
        interactive: bool = False
    ) -> Optional[Any]:
        """
        Plot cluster assignments in 2D space.

        Args:
            df: DataFrame with cluster assignments
            x_column: X-axis column name
            y_column: Y-axis column name
            cluster_column: Cluster assignment column
            title: Plot title
            show_centroids: Show cluster centroids
            centroids: Centroid coordinates
            save_name: Filename for saving
            interactive: Use plotly

        Returns:
            Figure object if interactive
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = px.scatter(
                df, x=x_column, y=y_column,
                color=cluster_column,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set1
            )

            if save_name:
                save_path = self.output_dir / f"{save_name}.html"
                fig.write_html(str(save_path))

            return fig

        fig, ax = plt.subplots(figsize=self.figsize)

        # Get unique clusters
        clusters = df[cluster_column].unique()
        colors = sns.color_palette(self.palette, len(clusters))

        for i, cluster in enumerate(sorted(clusters)):
            mask = df[cluster_column] == cluster
            ax.scatter(
                df.loc[mask, x_column],
                df.loc[mask, y_column],
                c=[colors[i]],
                label=f'Cluster {cluster}',
                alpha=0.6,
                s=50
            )

        # Plot centroids
        if show_centroids and centroids is not None:
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                c='black',
                marker='X',
                s=200,
                label='Centroids'
            )

        ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved cluster plot: {save_path}")

        plt.show()
        return None

    def plot_elbow(
        self,
        k_range: List[int],
        inertias: List[float],
        silhouettes: Optional[List[float]] = None,
        title: str = "Elbow Method for Optimal K",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot elbow curve for K-means cluster selection.

        Args:
            k_range: Range of K values tested
            inertias: Inertia values for each K
            silhouettes: Silhouette scores for each K
            title: Plot title
            save_name: Filename for saving
        """
        if silhouettes:
            (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))[1]
        else:
            ax1 = plt.subplots(figsize=self.figsize)[1]

        # Inertia plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Silhouette plot
        if silhouettes:
            ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax2.set_ylabel('Silhouette Score', fontsize=12)
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved elbow plot: {save_path}")

        plt.show()

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Plot title
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved confusion matrix: {save_path}")

        plt.show()

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: str = "ROC Curve",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: Area under curve
            title: Plot title
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved ROC curve: {save_path}")

        plt.show()

    def plot_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        bins: int = 50,
        kde: bool = True,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot distribution histogram with optional KDE.

        Args:
            df: Input DataFrame
            column: Column to plot
            title: Plot title
            bins: Number of histogram bins
            kde: Show kernel density estimate
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.histplot(
            data=df,
            x=column,
            bins=bins,
            kde=kde,
            ax=ax
        )

        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved distribution plot: {save_path}")

        plt.show()

    def plot_anomaly_scores(
        self,
        df: pd.DataFrame,
        score_column: str,
        threshold: float,
        date_column: Optional[str] = None,
        title: str = "Anomaly Scores",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot anomaly scores with threshold line.

        Args:
            df: DataFrame with anomaly scores
            score_column: Column containing scores
            threshold: Anomaly threshold
            date_column: Date column for x-axis
            title: Plot title
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if date_column:
            x = df[date_column]
        else:
            x = range(len(df))

        # Plot scores
        ax.scatter(x, df[score_column], c='blue', alpha=0.5, s=20)

        # Threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')

        # Highlight anomalies
        anomalies = df[df[score_column] < threshold]
        if date_column:
            ax.scatter(anomalies[date_column], anomalies[score_column],
                      c='red', s=50, label='Anomalies', zorder=5)
        else:
            ax.scatter(anomalies.index, anomalies[score_column],
                      c='red', s=50, label='Anomalies', zorder=5)

        ax.set_xlabel('Date' if date_column else 'Index', fontsize=12)
        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved anomaly scores plot: {save_path}")

        plt.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 20,
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Importance scores
            title: Plot title
            top_n: Number of top features to show
            save_name: Filename for saving
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.barh(
            range(len(indices)),
            importances[indices],
            color=sns.color_palette(self.palette)[0]
        )

        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance plot: {save_path}")

        plt.show()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        save_name: Optional[str] = None
    ) -> None:
        """
        Plot correlation matrix heatmap.

        Args:
            df: Input DataFrame
            columns: Columns to include (None for all numeric)
            title: Plot title
            save_name: Filename for saving
        """
        if columns:
            corr = df[columns].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()

        fig, ax = plt.subplots(figsize=self.figsize)

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation matrix: {save_path}")

        plt.show()

    def create_dashboard(
        self,
        plots: List[Dict[str, Any]],
        title: str = "Analytics Dashboard",
        save_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a multi-plot dashboard.

        Args:
            plots: List of plot configurations
            title: Dashboard title
            save_name: Filename for saving

        Returns:
            Plotly figure if available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return None

        n_plots = len(plots)
        rows = (n_plots + 1) // 2
        cols = min(2, n_plots)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[p.get('title', '') for p in plots]
        )

        for i, plot_config in enumerate(plots):
            # Add traces based on plot type
            # This is simplified - extend as needed
            pass  # Placeholder for future implementation

        fig.update_layout(
            title=title,
            showlegend=True,
            height=400 * rows
        )

        if save_name:
            save_path = self.output_dir / f"{save_name}.html"
            fig.write_html(str(save_path))
            logger.info(f"Saved dashboard: {save_path}")

        return fig
