"""
Stage B: Multi-Stage Feature Engineering

This package implements feature engineering components for the VMETHOD forex prediction framework:
1. Enhanced Feature Library - Technical indicators for trend, momentum, and volatility
2. Pattern Recognition System - Candlestick patterns, reversal points, and breakouts
3. Feature Selection Framework - Methods to identify the most predictive features
"""

from .feature_engineering import EnhancedFeatureLibrary
from .pattern_recognition import PatternRecognitionSystem
from .feature_selection import FeatureSelectionFramework

__all__ = ['EnhancedFeatureLibrary', 'PatternRecognitionSystem', 'FeatureSelectionFramework']