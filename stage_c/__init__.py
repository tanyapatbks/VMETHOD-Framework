"""
Stage C: Advanced Prediction Framework

This package implements model development and prediction components for the VMETHOD forex prediction framework:
1. Model Development - Implementation of LSTM, XGBoost, GRU, and TFT models
2. Quantile Regression System - Prediction intervals rather than single-point forecasts
3. Ensemble Framework - Methods to combine multiple models for more robust predictions
"""

from .model_development import ModelDevelopment, LSTMModel, XGBoostModel, GRUModel, TFTModel
from .quantile_regression import QuantileRegressionSystem
from .ensemble_framework import EnsembleFramework

__all__ = [
    'ModelDevelopment', 
    'LSTMModel', 
    'XGBoostModel', 
    'GRUModel', 
    'TFTModel', 
    'QuantileRegressionSystem', 
    'EnsembleFramework'
]