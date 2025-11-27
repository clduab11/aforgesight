"""
Isolation Forest Anomaly Detection Module
==========================================

Implements Isolation Forest for detecting fraudulent transactions
with support for ensemble methods and threshold optimization.

Usage:
    from src.fraud_detection import IsolationForestDetector

    detector = IsolationForestDetector()
    detector.fit(features)
    predictions = detector.predict(new_features)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detection for fraud identification.

    Features:
    - Multiple algorithm support (Isolation Forest, One-Class SVM, LOF)
    - Automatic threshold optimization
    - Ensemble predictions
    - Anomaly score calibration

    Example:
        >>> detector = IsolationForestDetector()
        >>> detector.fit(features_df, feature_columns)
        >>> predictions = detector.predict(new_data)
        >>> flagged = new_data[predictions == -1]
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        contamination: float = 0.01,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest Detector.

        Args:
            n_estimators: Number of isolation trees
            max_samples: Number of samples to draw
            contamination: Expected proportion of outliers
            max_features: Number of features to draw
            bootstrap: Bootstrap sampling
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.threshold = None

        # Alternative models
        self.ensemble_models = {}

        logger.info("IsolationForestDetector initialized")

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        scale_features: bool = True
    ) -> 'IsolationForestDetector':
        """
        Fit Isolation Forest model.

        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            scale_features: Whether to scale features

        Returns:
            Self for method chaining

        Example:
            >>> detector.fit(df, ['amount', 'hour', 'frequency'])
        """
        self.feature_columns = feature_columns

        # Extract features
        X = df[feature_columns].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X)

        # Calculate default threshold
        scores = self.model.decision_function(X)
        self.threshold = np.percentile(scores, self.contamination * 100)

        logger.info(f"Fitted Isolation Forest on {len(df)} samples")
        logger.info(f"Default threshold: {self.threshold:.4f}")

        return self

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal).

        Args:
            df: DataFrame with features
            threshold: Custom threshold (None for default)

        Returns:
            Array of predictions (-1: anomaly, 1: normal)

        Example:
            >>> predictions = detector.predict(new_data)
            >>> anomalies = new_data[predictions == -1]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if self.scaler:
            X = self.scaler.transform(X)

        if threshold is None:
            return self.model.predict(X)
        else:
            scores = self.model.decision_function(X)
            return np.where(scores < threshold, -1, 1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly probability scores.

        Args:
            df: DataFrame with features

        Returns:
            Array of anomaly probabilities (0-1, higher = more anomalous)

        Example:
            >>> probs = detector.predict_proba(data)
            >>> high_risk = data[probs > 0.8]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if self.scaler:
            X = self.scaler.transform(X)

        # Get decision scores
        scores = self.model.decision_function(X)

        # Convert to probability-like score (0-1)
        # More negative = more anomalous
        probs = 1 - (scores - scores.min()) / (scores.max() - scores.min())

        return probs

    def get_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get raw anomaly scores (more negative = more anomalous).

        Args:
            df: DataFrame with features

        Returns:
            Array of anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if self.scaler:
            X = self.scaler.transform(X)

        return self.model.decision_function(X)

    def fit_ensemble(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        methods: List[str] = ['isolation_forest', 'lof']
    ) -> 'IsolationForestDetector':
        """
        Fit ensemble of anomaly detection models.

        Args:
            df: DataFrame with features
            feature_columns: Feature column names
            methods: List of methods to use

        Returns:
            Self for method chaining

        Example:
            >>> detector.fit_ensemble(df, features, ['isolation_forest', 'lof', 'ocsvm'])
        """
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Fit each model
        if 'isolation_forest' in methods:
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X)
            self.ensemble_models['isolation_forest'] = self.model

        if 'lof' in methods:
            lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
            lof.fit(X)
            self.ensemble_models['lof'] = lof

        if 'ocsvm' in methods:
            ocsvm = OneClassSVM(
                kernel='rbf',
                nu=self.contamination,
                gamma='scale'
            )
            ocsvm.fit(X)
            self.ensemble_models['ocsvm'] = ocsvm

        logger.info(f"Fitted ensemble with {len(self.ensemble_models)} models")
        return self

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        voting: str = 'soft'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble predictions from multiple models.

        Args:
            df: DataFrame with features
            voting: 'hard' or 'soft' voting

        Returns:
            Tuple of (predictions, confidence_scores)

        Example:
            >>> predictions, confidence = detector.predict_ensemble(data)
        """
        if not self.ensemble_models:
            raise ValueError("Ensemble not fitted. Call fit_ensemble() first.")

        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X = self.scaler.transform(X)

        predictions = []

        for name, model in self.ensemble_models.items():
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if voting == 'hard':
            # Majority voting
            final_pred = np.sign(np.sum(predictions, axis=0))
            final_pred = np.where(final_pred == 0, -1, final_pred)
            confidence = np.abs(np.sum(predictions, axis=0)) / len(self.ensemble_models)
        else:
            # Average scores
            scores = []
            for name, model in self.ensemble_models.items():
                score = model.decision_function(X)
                # Normalize scores
                score = (score - score.min()) / (score.max() - score.min() + 1e-10)
                scores.append(score)

            avg_score = np.mean(scores, axis=0)
            final_pred = np.where(avg_score < 0.5, -1, 1)
            confidence = np.abs(avg_score - 0.5) * 2

        return final_pred.astype(int), confidence

    def optimize_threshold(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        metric: str = 'f1',
        n_thresholds: int = 100
    ) -> float:
        """
        Optimize detection threshold using labeled data.

        Args:
            df: DataFrame with features
            labels: True labels (1: normal, -1 or 0: fraud)
            metric: Optimization metric ('f1', 'precision', 'recall')
            n_thresholds: Number of thresholds to test

        Returns:
            Optimal threshold value

        Example:
            >>> optimal_threshold = detector.optimize_threshold(df, labels, metric='f1')
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        scores = self.get_anomaly_scores(df)

        # Convert labels to binary (0: normal, 1: fraud)
        y_true = np.where(labels == 1, 0, 1)

        # Test thresholds
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
        best_score = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            y_pred = np.where(scores < threshold, 1, 0)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.threshold = best_threshold
        logger.info(f"Optimized threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")

        return best_threshold

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Estimate feature importance using permutation importance.
        
        Note: Isolation Forest doesn't provide native feature importance.
        This method uses permutation importance on the validation set if available,
        otherwise returns an error message.

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not hasattr(self, 'X_val') or self.X_val is None or len(self.X_val) == 0:
            logger.warning("No validation data available. Cannot compute feature importance.")
            # Return empty dataframe with proper structure
            return pd.DataFrame(columns=['feature', 'importance'])

        # Use permutation importance
        from sklearn.inspection import permutation_importance
        
        # Get baseline score
        baseline_scores = self.model.score_samples(self.X_val)
        baseline_mean = np.mean(baseline_scores)
        
        importance_scores = []
        for i, feature in enumerate(self.feature_columns):
            # Permute feature
            X_permuted = self.X_val.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate score with permuted feature
            permuted_scores = self.model.score_samples(X_permuted)
            permuted_mean = np.mean(permuted_scores)
            
            # Importance is the decrease in score
            importance_scores.append({
                'feature': feature,
                'importance': abs(baseline_mean - permuted_mean)
            })

        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Normalize to sum to 1
        total = importance_df['importance'].sum()
        if total > 0:
            importance_df['importance'] = importance_df['importance'] / total

        return importance_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        return {
            'type': 'Isolation Forest',
            'n_estimators': self.n_estimators,
            'contamination': self.contamination,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'threshold': self.threshold,
            'n_features': len(self.feature_columns) if self.feature_columns else 0
        }

    def flag_transactions(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        score_column: str = 'anomaly_score',
        flag_column: str = 'is_flagged'
    ) -> pd.DataFrame:
        """
        Flag transactions as potential fraud.

        Args:
            df: DataFrame with features
            threshold: Detection threshold
            score_column: Name for score column
            flag_column: Name for flag column

        Returns:
            DataFrame with scores and flags

        Example:
            >>> flagged_df = detector.flag_transactions(transactions)
            >>> frauds = flagged_df[flagged_df['is_flagged']]
        """
        result = df.copy()

        # Get scores
        scores = self.get_anomaly_scores(df)
        result[score_column] = scores

        # Get predictions
        if threshold is None:
            predictions = self.predict(df)
        else:
            predictions = self.predict(df, threshold=threshold)

        result[flag_column] = predictions == -1

        # Add probability
        result['fraud_probability'] = self.predict_proba(df)

        # Risk level
        result['risk_level'] = pd.cut(
            result['fraud_probability'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        n_flagged = result[flag_column].sum()
        logger.info(f"Flagged {n_flagged} transactions ({n_flagged/len(df)*100:.2f}%)")

        return result

    def get_recommendations(
        self,
        flagged_df: pd.DataFrame,
        score_column: str = 'anomaly_score'
    ) -> List[str]:
        """
        Generate recommendations based on detection results.

        Args:
            flagged_df: DataFrame with flagged transactions
            score_column: Anomaly score column

        Returns:
            List of recommendation strings
        """
        recommendations = []

        n_flagged = flagged_df['is_flagged'].sum() if 'is_flagged' in flagged_df.columns else 0
        total = len(flagged_df)
        flag_rate = n_flagged / total * 100 if total > 0 else 0

        # Basic recommendations
        if flag_rate > 5:
            recommendations.append(
                f"High flag rate ({flag_rate:.1f}%): Consider reviewing threshold settings "
                "to reduce false positives."
            )
        elif flag_rate < 0.1:
            recommendations.append(
                f"Very low flag rate ({flag_rate:.2f}%): Consider lowering threshold "
                "to catch more potential fraud."
            )

        # Score distribution
        if score_column in flagged_df.columns:
            scores = flagged_df[score_column]
            if scores.std() < 0.1:
                recommendations.append(
                    "Low score variance: Consider adding more diverse features "
                    "to improve detection sensitivity."
                )

        # Risk level distribution
        if 'risk_level' in flagged_df.columns:
            critical = (flagged_df['risk_level'] == 'Critical').sum()
            if critical > total * 0.01:
                recommendations.append(
                    f"{critical} critical-risk transactions detected. "
                    "Recommend immediate manual review."
                )

        # General recommendations
        recommendations.extend([
            "Implement real-time scoring for immediate fraud prevention.",
            "Set up automated alerts for critical-risk transactions.",
            "Regularly retrain model with new fraud patterns.",
            "Consider ensemble methods for improved accuracy."
        ])

        return recommendations
