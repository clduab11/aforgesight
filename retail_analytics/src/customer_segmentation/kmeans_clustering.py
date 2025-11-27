"""
K-Means Clustering Module
=========================

Implements K-Means++ clustering with automatic cluster selection
and comprehensive analysis for customer segmentation.

Usage:
    from src.customer_segmentation import KMeansSegmenter

    segmenter = KMeansSegmenter()
    segmenter.fit(features, ['recency', 'frequency', 'monetary'])
    labels = segmenter.predict(new_features)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from loguru import logger
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


class KMeansSegmenter:
    """
    K-Means++ clustering for customer segmentation.

    Features:
    - Automatic optimal K selection via elbow/silhouette
    - Multiple scaling methods
    - PCA/t-SNE for visualization
    - Comprehensive cluster profiling

    Example:
        >>> segmenter = KMeansSegmenter()
        >>> segmenter.fit(df, ['recency', 'frequency', 'monetary'])
        >>> labels = segmenter.predict(new_data)
        >>> profiles = segmenter.get_cluster_profiles(df)
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_clusters_range: Tuple[int, int] = (2, 10),
        init: str = 'k-means++',
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
        scaling_method: str = 'standard'
    ):
        """
        Initialize K-Means Segmenter.

        Args:
            n_clusters: Number of clusters (None for auto-selection)
            n_clusters_range: Range for automatic K selection
            init: Initialization method ('k-means++' or 'random')
            n_init: Number of initializations
            max_iter: Maximum iterations
            random_state: Random seed for reproducibility
            scaling_method: Feature scaling method ('standard', 'minmax', 'robust')
        """
        self.n_clusters = n_clusters
        self.n_clusters_range = n_clusters_range
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaling_method = scaling_method

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        # For cluster selection
        self.k_range = None
        self.inertias = []
        self.silhouettes = []

        logger.info("KMeansSegmenter initialized")

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        auto_select_k: bool = True
    ) -> 'KMeansSegmenter':
        """
        Fit K-Means model to data.

        Args:
            df: DataFrame with customer features
            feature_columns: List of feature column names
            auto_select_k: Automatically select optimal K

        Returns:
            Self for method chaining

        Example:
            >>> segmenter.fit(df, ['recency', 'frequency', 'monetary'])
        """
        self.feature_columns = feature_columns

        # Extract and scale features
        X = df[feature_columns].values

        # Handle missing values
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

        # Scale features
        self.scaler = self._get_scaler()
        X_scaled = self.scaler.fit_transform(X)

        # Auto-select K if not specified
        if self.n_clusters is None and auto_select_k:
            self.n_clusters = self._select_optimal_k(X_scaled)
            logger.info(f"Auto-selected K={self.n_clusters}")

        elif self.n_clusters is None:
            self.n_clusters = 4  # Default

        # Fit K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        self.labels_ = self.model.fit_predict(X_scaled)
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_

        logger.info(f"Fitted K-Means with {self.n_clusters} clusters")
        logger.info(f"Inertia: {self.inertia_:.2f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Args:
            df: DataFrame with features

        Returns:
            Array of cluster labels

        Example:
            >>> labels = segmenter.predict(new_customers)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].values

        # Handle missing values
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            df: DataFrame with features
            feature_columns: Feature column names

        Returns:
            Array of cluster labels
        """
        self.fit(df, feature_columns)
        return self.labels_

    def _get_scaler(self):
        """Get appropriate scaler based on method."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def _select_optimal_k(self, X: np.ndarray) -> int:
        """Select optimal K using elbow and silhouette methods."""
        self.k_range = range(self.n_clusters_range[0], self.n_clusters_range[1] + 1)
        self.inertias = []
        self.silhouettes = []

        for k in self.k_range:
            kmeans = KMeans(
                n_clusters=k,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            labels = kmeans.fit_predict(X)

            self.inertias.append(kmeans.inertia_)
            self.silhouettes.append(silhouette_score(X, labels))

        # Find elbow point
        elbow_k = self._find_elbow(list(self.k_range), self.inertias)

        # Find best silhouette
        best_silhouette_idx = np.argmax(self.silhouettes)
        silhouette_k = list(self.k_range)[best_silhouette_idx]

        # Use average of both methods (rounded)
        optimal_k = int(round((elbow_k + silhouette_k) / 2))

        logger.info(f"Elbow method K={elbow_k}, Silhouette K={silhouette_k}")
        return optimal_k

    def _find_elbow(self, k_range: List[int], inertias: List[float]) -> int:
        """Find elbow point using the knee/elbow method."""
        # Calculate the angle at each point
        all_coords = np.vstack([k_range, inertias]).T

        # Vector from first to last point
        first_point = all_coords[0]
        last_point = all_coords[-1]
        line_vec = last_point - first_point

        # Normalize
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

        # Distance from each point to the line
        vec_from_first = all_coords - first_point
        scalar_proj = np.dot(vec_from_first, line_vec_norm)
        vec_from_line = vec_from_first - np.outer(scalar_proj, line_vec_norm)
        distances = np.sqrt(np.sum(vec_from_line**2, axis=1))

        # Return K with maximum distance
        elbow_idx = np.argmax(distances)
        return k_range[elbow_idx]

    def get_cluster_metrics(self, df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            df: DataFrame with features (uses fitted data if None)

        Returns:
            Dictionary of clustering metrics

        Example:
            >>> metrics = segmenter.get_cluster_metrics()
            >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if df is not None:
            X_scaled = self.scaler.transform(df[self.feature_columns].values)
            labels = self.model.predict(X_scaled)
        else:
            X_scaled = self.model.cluster_centers_
            labels = self.labels_

        # Reconstruct scaled data for metrics
        # For simplicity, we'll use the current labels
        if df is not None:
            X_for_metrics = X_scaled
        else:
            # Need original data - this is a limitation
            logger.warning("Using cluster centers for metrics. Pass df for accurate metrics.")
            return {
                'n_clusters': self.n_clusters,
                'inertia': self.inertia_
            }

        metrics = {
            'n_clusters': self.n_clusters,
            'inertia': self.inertia_,
            'silhouette_score': silhouette_score(X_for_metrics, labels),
            'calinski_harabasz': calinski_harabasz_score(X_for_metrics, labels),
            'davies_bouldin': davies_bouldin_score(X_for_metrics, labels)
        }

        return metrics

    def get_cluster_profiles(
        self,
        df: pd.DataFrame,
        include_all_features: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate detailed profiles for each cluster.

        Args:
            df: DataFrame with features and cluster labels
            include_all_features: Include all numeric features in profile

        Returns:
            Dictionary of cluster profiles

        Example:
            >>> profiles = segmenter.get_cluster_profiles(df)
            >>> for cluster, profile in profiles.items():
            ...     print(f"Cluster {cluster}: {profile['size']} customers")
        """
        if self.labels_ is None:
            raise ValueError("No labels available. Call fit() first.")

        df = df.copy()
        df['cluster'] = self.labels_

        profiles = {}

        for cluster in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster]

            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }

            # Feature statistics
            if include_all_features:
                numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
                feature_cols = [c for c in numeric_cols if c != 'cluster']
            else:
                feature_cols = self.feature_columns

            for col in feature_cols:
                if col in cluster_data.columns:
                    profile[f'avg_{col}'] = cluster_data[col].mean()
                    profile[f'median_{col}'] = cluster_data[col].median()
                    profile[f'std_{col}'] = cluster_data[col].std()

            profiles[cluster] = profile

        return profiles

    def get_cluster_centers_original(self) -> pd.DataFrame:
        """
        Get cluster centers in original (unscaled) feature space.

        Returns:
            DataFrame with cluster centers
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centers = self.scaler.inverse_transform(self.cluster_centers_)

        return pd.DataFrame(
            centers,
            columns=self.feature_columns,
            index=[f'Cluster_{i}' for i in range(self.n_clusters)]
        )

    def reduce_dimensions(
        self,
        df: pd.DataFrame,
        method: str = 'pca',
        n_components: int = 2
    ) -> pd.DataFrame:
        """
        Reduce dimensions for visualization.

        Args:
            df: DataFrame with features
            method: Reduction method ('pca' or 'tsne')
            n_components: Number of output components

        Returns:
            DataFrame with reduced dimensions

        Example:
            >>> reduced = segmenter.reduce_dimensions(df, method='pca')
            >>> # Plot with cluster colors
        """
        X = df[self.feature_columns].values

        # Handle missing values
        if np.isnan(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)

        X_scaled = self.scaler.transform(X)

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            components = reducer.fit_transform(X_scaled)
            col_names = [f'PC{i+1}' for i in range(n_components)]

            # Log explained variance
            logger.info(f"PCA explained variance: {reducer.explained_variance_ratio_}")

        elif method == 'tsne':
            if not TSNE_AVAILABLE:
                raise ImportError("t-SNE not available")

            reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=min(30, len(X) - 1)
            )
            components = reducer.fit_transform(X_scaled)
            col_names = [f'tSNE{i+1}' for i in range(n_components)]

        else:
            raise ValueError(f"Unknown method: {method}")

        result = pd.DataFrame(components, columns=col_names, index=df.index)

        # Add cluster labels if available
        if self.labels_ is not None and len(self.labels_) == len(df):
            result['cluster'] = self.labels_

        return result

    def get_selection_plot_data(self) -> Dict[str, Any]:
        """
        Get data for plotting K selection (elbow/silhouette).

        Returns:
            Dictionary with plot data
        """
        if not self.inertias:
            raise ValueError("No selection data. Call fit() with auto_select_k=True")

        return {
            'k_range': list(self.k_range),
            'inertias': self.inertias,
            'silhouettes': self.silhouettes,
            'selected_k': self.n_clusters
        }

    def assign_cluster_names(
        self,
        profiles: Dict[int, Dict[str, Any]],
        naming_strategy: str = 'value'
    ) -> Dict[int, str]:
        """
        Automatically assign descriptive names to clusters.

        Args:
            profiles: Cluster profiles from get_cluster_profiles()
            naming_strategy: Strategy for naming ('value', 'rfm', 'custom')

        Returns:
            Dictionary mapping cluster IDs to names
        """
        names = {}

        if naming_strategy == 'value':
            # Sort clusters by monetary value
            sorted_clusters = sorted(
                profiles.items(),
                key=lambda x: x[1].get('avg_monetary', 0),
                reverse=True
            )

            value_labels = [
                'Premium', 'High Value', 'Medium Value',
                'Low Value', 'Budget', 'Minimal'
            ]

            for i, (cluster_id, _) in enumerate(sorted_clusters):
                if i < len(value_labels):
                    names[cluster_id] = value_labels[i]
                else:
                    names[cluster_id] = f'Segment_{cluster_id}'

        elif naming_strategy == 'rfm':
            for cluster_id, profile in profiles.items():
                r = profile.get('avg_recency', 0)
                f = profile.get('avg_frequency', 0)
                m = profile.get('avg_monetary', 0)

                # Simple classification
                if m > np.median([p.get('avg_monetary', 0) for p in profiles.values()]):
                    if f > np.median([p.get('avg_frequency', 0) for p in profiles.values()]):
                        names[cluster_id] = 'Champions'
                    else:
                        names[cluster_id] = 'Big Spenders'
                elif f > np.median([p.get('avg_frequency', 0) for p in profiles.values()]):
                    names[cluster_id] = 'Loyal'
                elif r < np.median([p.get('avg_recency', 0) for p in profiles.values()]):
                    names[cluster_id] = 'Recent'
                else:
                    names[cluster_id] = 'At Risk'

        else:
            for cluster_id in profiles.keys():
                names[cluster_id] = f'Segment_{cluster_id}'

        return names

    def get_campaign_recommendations(
        self,
        profiles: Dict[int, Dict[str, Any]]
    ) -> List[str]:
        """
        Generate campaign recommendations for each cluster.

        Args:
            profiles: Cluster profiles

        Returns:
            List of recommendations for each cluster
        """
        recommendations = []

        # Sort by value
        sorted_clusters = sorted(
            profiles.items(),
            key=lambda x: x[1].get('avg_monetary', 0),
            reverse=True
        )

        for i, (cluster_id, profile) in enumerate(sorted_clusters):
            size = profile.get('size', 0)
            pct = profile.get('percentage', 0)
            avg_monetary = profile.get('avg_monetary', 0)
            avg_frequency = profile.get('avg_frequency', 0)
            avg_recency = profile.get('avg_recency', 0)

            if i == 0:  # Top cluster
                rec = f"Cluster {cluster_id} ({size} customers, {pct:.1f}%): "
                rec += "VIP treatment - exclusive offers, early access, loyalty rewards. "
                rec += f"High value (${avg_monetary:.2f} avg), focus on retention."
            elif avg_frequency > np.median([p.get('avg_frequency', 0) for p in profiles.values()]):
                rec = f"Cluster {cluster_id} ({size} customers, {pct:.1f}%): "
                rec += "Cross-sell and upsell opportunities. "
                rec += f"Frequent buyers ({avg_frequency:.1f} purchases), increase basket size."
            elif avg_recency < np.median([p.get('avg_recency', 0) for p in profiles.values()]):
                rec = f"Cluster {cluster_id} ({size} customers, {pct:.1f}%): "
                rec += "Nurture new relationships. "
                rec += f"Recent customers ({avg_recency:.0f} days), welcome series and education."
            else:
                rec = f"Cluster {cluster_id} ({size} customers, {pct:.1f}%): "
                rec += "Reactivation campaigns needed. "
                rec += f"At risk ({avg_recency:.0f} days since purchase), win-back offers."

            recommendations.append(rec)

        return recommendations
