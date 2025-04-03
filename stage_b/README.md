# VMETHOD Framework - Stage B: Multi-Stage Feature Engineering

This directory contains the implementation of Stage B of the VMETHOD framework, which focuses on multi-stage feature engineering for forex market prediction.

## Overview

Stage B builds upon the processed data from Stage A and consists of three main components:

1. **Enhanced Feature Library**
   - Calculation of technical indicators for trend, momentum, and volatility
   - Comprehensive set of indicators such as moving averages, MACD, RSI, Bollinger Bands, etc.
   - Visualization utilities for all indicators

2. **Pattern Recognition System**
   - Detection of candlestick patterns (Doji, Hammer, Engulfing, etc.)
   - Recognition of technical reversal points (Head and Shoulders, Double Tops/Bottoms, etc.)
   - Identification of breakout patterns (volume-confirmed, volatility-based, channel breakouts)

3. **Feature Selection Framework**
   - Multiple feature importance methods (Random Forest, Boruta, SHAP, Autoencoder)
   - Fusion system to combine results from different methods
   - Visualization and analysis of feature importance

## Implementation Details

### Enhanced Feature Library (`feature_engineering.py`)

The feature library implements three categories of technical indicators:

#### Trend Indicators
- **Moving Averages (SMA/EMA)**: Averages prices over different time periods (5, 10, 20, 50, 200)
- **MACD (Moving Average Convergence Divergence)**: Measures the relationship between two EMAs
- **DMI (Directional Movement Index)**: Measures the strength of a trend

#### Momentum Indicators
- **RSI (Relative Strength Index)**: Compares the magnitude of recent gains to recent losses
- **Stochastic Oscillator**: Compares current price to its range over a specific period
- **ROC (Rate of Change)**: Percentage change in price over a defined period
- **CCI (Commodity Channel Index)**: Identifies cyclical trends in price movements

#### Volatility Indicators
- **Bollinger Bands**: Moving average with bands at standard deviation levels
- **ATR (Average True Range)**: Measures market volatility
- **Keltner Channels**: Volatility-based bands around an EMA

### Pattern Recognition System (`pattern_recognition.py`)

The pattern recognition system identifies significant price patterns in three categories:

#### Candlestick Patterns
- **Doji**: Sessions where opening and closing prices are nearly identical
- **Hammer/Shooting Star**: Patterns with long shadows and small bodies
- **Engulfing Patterns**: When one candle's body completely engulfs the previous candle
- **Morning/Evening Star**: Three-candle patterns indicating potential reversals

#### Technical Reversal Patterns
- **Head and Shoulders**: A three-peak pattern often preceding trend reversals
- **Double Tops/Bottoms**: Two-peak or two-valley patterns suggesting resistance/support
- **Support and Resistance Levels**: Price levels where markets historically reverse
- **Fibonacci Retracement Levels**: Key levels based on the Fibonacci sequence

#### Breakout Detection
- **Volume-Confirmed Breakouts**: Price movements beyond ranges accompanied by increased volume
- **Volatility Expansion Breakouts**: Sudden increases in price range after periods of contraction
- **Channel Breakouts**: Prices moving beyond established channel boundaries

### Feature Selection Framework (`feature_selection.py`)

The feature selection framework employs multiple methods to identify the most predictive features:

#### Random Forest Feature Importance
- Uses the built-in feature importance from Random Forest models
- Ranks features by their contribution to prediction accuracy
- Provides threshold-based selection

#### Boruta Algorithm Implementation
- Extends Random Forest selection through an all-relevant feature approach
- Creates shadow features by randomizing original features
- Compares real feature importance to shadow feature importance

#### SHAP Value Analysis
- Calculates SHapley Additive exPlanations (SHAP) values for each feature
- Provides interpretable and fair feature importance
- Explains both feature importance and impact direction

#### Autoencoder Feature Selection
- Trains a neural network autoencoder to compress and reconstruct the feature space
- Extracts the compressed representation from the bottleneck layer
- Uses reconstruction weights to assess feature importance

#### Feature Fusion System
- Combines results from all selection methods
- Implements a voting mechanism to identify consensus important features
- Creates a robust final feature subset based on feature agreement across methods

## Usage

The Stage B components can be run either directly through the main script or as standalone modules:

```python
# Through main.py
python main.py --stage b

# As standalone modules
from stage_b.feature_engineering import EnhancedFeatureLibrary
from stage_b.pattern_recognition import PatternRecognitionSystem
from stage_b.feature_selection import FeatureSelectionFramework

# Initialize and use components
feature_library = EnhancedFeatureLibrary()
feature_data = feature_library.calculate_all_features(preprocessed_data)
```

## Configuration

The behavior of Stage B components can be customized through the `config.yaml` file, which includes sections for:

- Feature engineering settings (indicator parameters)
- Pattern recognition thresholds
- Feature selection method parameters
- Feature fusion criteria

## Outputs

Stage B generates several outputs:

1. **Feature Data**: CSV files with calculated technical indicators
2. **Pattern Data**: CSV files with detected price patterns
3. **Selected Features**: CSV files with the most predictive features
4. **Visualizations**: Various charts for indicators, patterns, and feature importance

These outputs provide a rich set of features for prediction model development in Stage C.