# VMETHOD Framework - Stage C: Advanced Prediction Framework

This directory contains the implementation of Stage C of the VMETHOD framework, which focuses on advanced prediction models for forex market forecasting.

## Overview

Stage C builds upon the preprocessed data from Stage A and the engineered features from Stage B, implementing sophisticated prediction models and ensemble techniques. The stage consists of three main components:

1. **Model Development**
   - Implementation of four base prediction models: LSTM, XGBoost, GRU, and TFT
   - Support for both single-pair and bagging approaches
   - Comprehensive performance evaluation methodology

2. **Quantile Regression System**
   - Prediction intervals instead of single-point forecasts
   - Uncertainty quantification for market predictions
   - Model calibration assessment

3. **Ensemble Framework**
   - Multiple ensemble methods (weighted average, stacking, dynamic weighting)
   - Regime-specific model selection
   - Ensemble diversity analysis

## Implementation Details

### Model Development (`model_development.py`)

The model development module implements four state-of-the-art prediction models:

#### LSTM (Long Short-Term Memory)
- **Architecture**: Sequential LSTM layers with dropout
- **Key Features**: Memory cells for capturing long-term dependencies, ability to remember relevant past information
- **Strengths**: Excellent at capturing complex temporal patterns and relationships

#### XGBoost (eXtreme Gradient Boosting)
- **Architecture**: Gradient boosting with decision trees
- **Key Features**: Advanced regularization, tree pruning, parallel processing
- **Strengths**: Handles non-linear relationships and feature interactions very well

#### GRU (Gated Recurrent Unit)
- **Architecture**: Bidirectional GRU with attention mechanism
- **Key Features**: Update and reset gates to control information flow
- **Strengths**: Simpler than LSTM but nearly as effective, faster training

#### TFT (Temporal Fusion Transformer)
- **Architecture**: Transformer architecture with temporal components
- **Key Features**: Multi-head attention, variable selection, specialized temporal processing
- **Strengths**: State-of-the-art performance on time series problems, attention to relevant historical points

The module also includes:
- Base model class with common functionality for all models
- Standardized preprocessing and evaluation methods
- Comprehensive visualization tools for model analysis
- Performance comparison between single-pair and bagging approaches

### Quantile Regression System (`quantile_regression.py`)

The quantile regression system provides prediction intervals rather than single-point forecasts:

- **Multiple Quantile Prediction**: Forecasts 10th, 25th, 50th (median), 75th, and 90th percentiles
- **Two Implementation Methods**:
  - Neural network with custom quantile loss function
  - XGBoost with quantile regression objective
- **Calibration Assessment**: Evaluates whether the predicted quantiles match their theoretical probabilities
- **Visualizations**:
  - Prediction intervals showing uncertainty bounds
  - Calibration plots showing model reliability

This approach acknowledges the inherent uncertainty in financial markets, providing traders with a range of likely outcomes rather than a single forecast.

### Ensemble Framework (`ensemble_framework.py`)

The ensemble framework combines multiple models to create more robust predictions:

- **Weighted Average Ensemble**: Combines predictions with fixed weights
- **Stacking Ensemble**: Uses a meta-model to learn optimal combination weights
- **Dynamic Weighting**: Adjusts weights based on recent model performance
- **Regime-Specific Ensemble**: Uses different model combinations for different market conditions

Advanced features include:
- Ensemble diversity analysis to measure model complementarity
- Meta-model training with cross-validation
- Visualization of model weights and importance
- Dynamic adaptation to changing market conditions

## Usage

The Stage C components can be run either through the main script or as standalone modules:

```python
# Through main.py
python main.py --stage c

# As standalone modules
from stage_c.model_development import ModelDevelopment, LSTMModel, XGBoostModel, GRUModel, TFTModel
from stage_c.quantile_regression import QuantileRegressionSystem
from stage_c.ensemble_framework import EnsembleFramework

# Initialize and use components
model_dev = ModelDevelopment(models_dir='models/')
model_dev.train_single_approach(single_datasets, currency_pairs)
```

## Configuration

The behavior of Stage C components can be customized through the `config.yaml` file, which includes sections for:

- Model development settings (architecture, hyperparameters)
- Quantile regression parameters
- Ensemble methods and configurations
- Training and evaluation settings

## Outputs

Stage C generates several outputs:

1. **Trained Models**: Saved model files for all trained architectures
2. **Performance Reports**: Detailed evaluation metrics for all models
3. **Prediction Intervals**: Quantile forecasts showing uncertainty bounds
4. **Ensemble Weights**: Model combination weights for different approaches
5. **Visualizations**: Performance comparisons, calibration plots, and ensemble analysis

These outputs provide a comprehensive prediction framework for forex trading, with robust uncertainty quantification and adaptive model selection.

## Integration with the Framework

Stage C integrates with the previous stages of the VMETHOD framework:

- Takes feature-engineered data from Stage B as input
- Leverages market regime information from Stage A for regime-specific ensembles
- Provides predictions with confidence intervals for trading decisions

The stage implements a flexible framework that can be extended with additional models and ensemble techniques as needed.