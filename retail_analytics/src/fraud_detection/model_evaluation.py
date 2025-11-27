"""
Fraud Detection Model Evaluation Module
=======================================

Comprehensive evaluation metrics for anomaly detection models
including precision, recall, ROC curves, and business metrics.

Usage:
    from src.fraud_detection import FraudEvaluator

    evaluator = FraudEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_scores)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class FraudEvaluator:
    """
    Comprehensive evaluation toolkit for fraud detection models.

    Provides metrics calculation, threshold analysis, and
    business impact assessment.

    Example:
        >>> evaluator = FraudEvaluator()
        >>> metrics = evaluator.evaluate(y_true, y_pred, y_scores)
        >>> print(f"Precision: {metrics['precision']:.3f}")
    """

    def __init__(self):
        """Initialize FraudEvaluator."""
        logger.info("FraudEvaluator initialized")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        pos_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels (1: fraud, 0: normal)
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional)
            pos_label: Label for positive class (fraud)

        Returns:
            Dictionary of evaluation metrics

        Example:
            >>> metrics = evaluator.evaluate(y_true, y_pred, y_scores)
        """
        # Ensure proper format
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Convert -1/1 to 0/1 if needed
        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)
        if -1 in y_pred:
            y_pred = np.where(y_pred == -1, 1, 0)

        # Basic metrics
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': np.mean(y_true == y_pred)
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

            # Derived metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Score-based metrics
        if y_scores is not None:
            y_scores = np.array(y_scores).flatten()

            # Ensure scores are in right direction (higher = more anomalous)
            if np.corrcoef(y_scores, y_true)[0, 1] < 0:
                y_scores = -y_scores

            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                metrics['avg_precision'] = average_precision_score(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC: {e}")
                metrics['auc_roc'] = None
                metrics['avg_precision'] = None

        # Distribution metrics
        metrics['fraud_rate_actual'] = y_true.mean()
        metrics['fraud_rate_predicted'] = y_pred.mean()
        metrics['total_samples'] = len(y_true)
        metrics['total_frauds'] = y_true.sum()

        return metrics

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        # Convert if needed
        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)
        if -1 in y_pred:
            y_pred = np.where(y_pred == -1, 1, 0)

        return classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fraud'],
            digits=4
        )

    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.

        Args:
            y_true: True labels
            y_scores: Anomaly scores

        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()

        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)

        # Ensure higher score = more likely fraud
        if np.corrcoef(y_scores, y_true)[0, 1] < 0:
            y_scores = -y_scores

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return fpr, tpr, thresholds

    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.

        Args:
            y_true: True labels
            y_scores: Anomaly scores

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()

        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)

        if np.corrcoef(y_scores, y_true)[0, 1] < 0:
            y_scores = -y_scores

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        return precision, recall, thresholds

    def analyze_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_thresholds: int = 100
    ) -> pd.DataFrame:
        """
        Analyze performance across different thresholds.

        Args:
            y_true: True labels
            y_scores: Anomaly scores
            n_thresholds: Number of thresholds to test

        Returns:
            DataFrame with metrics at each threshold
        """
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()

        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)

        thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
        results = []

        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)

            results.append({
                'threshold': threshold,
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'flagged_rate': y_pred.mean()
            })

        return pd.DataFrame(results)

    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
        investigation_cost: float = 10.0,
        fraud_cost_ratio: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate business-oriented metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            amounts: Transaction amounts
            investigation_cost: Cost per investigation
            fraud_cost_ratio: Fraud amount multiplier for loss

        Returns:
            Dictionary of business metrics

        Example:
            >>> business = evaluator.calculate_business_metrics(
            ...     y_true, y_pred, amounts,
            ...     investigation_cost=10, fraud_cost_ratio=1.5
            ... )
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        amounts = np.array(amounts).flatten()

        if -1 in y_true:
            y_true = np.where(y_true == -1, 1, 0)
        if -1 in y_pred:
            y_pred = np.where(y_pred == -1, 1, 0)

        # Confusion matrix components
        tp = ((y_true == 1) & (y_pred == 1))
        fp = ((y_true == 0) & (y_pred == 1))
        fn = ((y_true == 1) & (y_pred == 0))

        # Amounts
        caught_fraud_amount = amounts[tp].sum()
        missed_fraud_amount = amounts[fn].sum()
        total_fraud_amount = amounts[y_true == 1].sum()

        # Costs
        investigation_costs = (tp.sum() + fp.sum()) * investigation_cost
        fraud_loss = missed_fraud_amount * fraud_cost_ratio
        prevented_loss = caught_fraud_amount * fraud_cost_ratio

        metrics = {
            'caught_fraud_amount': caught_fraud_amount,
            'missed_fraud_amount': missed_fraud_amount,
            'total_fraud_amount': total_fraud_amount,
            'fraud_detection_rate': caught_fraud_amount / (total_fraud_amount + 1e-10),
            'investigation_costs': investigation_costs,
            'fraud_loss': fraud_loss,
            'prevented_loss': prevented_loss,
            'net_savings': prevented_loss - investigation_costs,
            'roi': (prevented_loss - investigation_costs) / (investigation_costs + 1e-10)
        }

        return metrics

    def find_optimal_threshold_business(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
        investigation_cost: float = 10.0,
        fraud_cost_ratio: float = 1.0,
        n_thresholds: int = 100
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold that maximizes net savings.

        Args:
            y_true: True labels
            y_scores: Anomaly scores
            amounts: Transaction amounts
            investigation_cost: Cost per investigation
            fraud_cost_ratio: Fraud amount multiplier
            n_thresholds: Number of thresholds to test

        Returns:
            Tuple of (optimal_threshold, best_metrics)
        """
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()

        thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)

        best_savings = float('-inf')
        best_threshold = thresholds[0]
        best_metrics = {}

        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)

            metrics = self.calculate_business_metrics(
                y_true, y_pred, amounts,
                investigation_cost, fraud_cost_ratio
            )

            if metrics['net_savings'] > best_savings:
                best_savings = metrics['net_savings']
                best_threshold = threshold
                best_metrics = metrics

        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"Net savings: ${best_savings:,.2f}")

        return best_threshold, best_metrics

    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            amounts: Transaction amounts

        Returns:
            Dictionary with complete evaluation
        """
        report = {
            'metrics': self.evaluate(y_true, y_pred, y_scores),
            'classification_report': self.get_classification_report(y_true, y_pred)
        }

        if y_scores is not None:
            report['threshold_analysis'] = self.analyze_thresholds(
                y_true, y_scores
            ).to_dict('records')

        if amounts is not None:
            report['business_metrics'] = self.calculate_business_metrics(
                y_true, y_pred, amounts
            )

        return report

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Compare multiple fraud detection models.

        Args:
            model_results: Dict of model_name -> {y_true, y_pred, y_scores}

        Returns:
            Comparison DataFrame
        """
        comparisons = []

        for model_name, results in model_results.items():
            y_true = results['y_true']
            y_pred = results['y_pred']
            y_scores = results.get('y_scores')

            metrics = self.evaluate(y_true, y_pred, y_scores)

            row = {
                'model': model_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc_roc': metrics.get('auc_roc', np.nan),
                'false_positive_rate': metrics.get('false_positive_rate', np.nan)
            }

            comparisons.append(row)

        df = pd.DataFrame(comparisons)
        df = df.sort_values('f1', ascending=False)

        return df
