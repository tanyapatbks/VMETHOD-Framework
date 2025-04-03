"""
Stage A: Single/Bagging Forecasting Implementation

This package handles data acquisition, preprocessing, and training strategy implementation
for the VMETHOD forex prediction framework.
"""

from .data_acquisition import DataAcquisitionSystem
from .preprocessing import AdvancedPreprocessingPipeline
from .training_strategy import TrainingStrategy

__all__ = ['DataAcquisitionSystem', 'AdvancedPreprocessingPipeline', 'TrainingStrategy']