"""
Customer Segmentation Module
============================

K-Means clustering with RFM feature engineering for customer segmentation.
"""

from .rfm_features import RFMFeatureEngineer
from .kmeans_clustering import KMeansSegmenter
from .segment_analysis import SegmentAnalyzer

__all__ = ["RFMFeatureEngineer", "KMeansSegmenter", "SegmentAnalyzer"]
