# VMETHOD: A Comprehensive Framework for Forex Market Prediction

This repository contains the implementation of the VMETHOD framework, a systematic approach to forex market prediction that combines statistical methods, signal processing techniques, and machine learning.

## Overview

The VMETHOD framework consists of four main stages:

1. **Stage A: Single/Bagging Forecasting Implementation**
   - Data acquisition, validation, and preprocessing
   - Advanced preprocessing techniques (fractional differentiation, wavelet transformation)
   - Training strategy implementation

2. **Stage B: Multi-Stage Feature Engineering** ✅
   - Technical indicator calculation (trend, momentum, volatility indicators)
   - Pattern recognition (candlestick patterns, reversal points, breakouts)
   - Feature selection (Random Forest, Boruta, SHAP, Autoencoder, Fusion) 

3. **Stage C: Advanced Prediction Framework** (planned)
   - Model development (LSTM, XGBoost, GRU, TFT)
   - Quantile regression
   - Ensemble methods

4. **Stage D: Comprehensive Evaluation** (planned)
   - Statistical metrics
   - Financial performance metrics
   - Trading simulation

## Project Structure

```
vmethod/
├── config/              # Configuration files
│   └── config.yaml      # Main configuration file
├── data/                # Data directory
│   ├── raw/             # Raw data files (your CSV files)
│   ├── processed/       # Processed data files
│   ├── training/        # Training-ready datasets
│   ├── features/        # Feature engineering outputs
│   ├── patterns/        # Pattern recognition outputs
│   └── selected_features/ # Feature selection outputs
├── results/             # Results and outputs
│   ├── figures/         # Figures and plots
│   │   ├── features/    # Technical indicator visualizations
│   │   ├── patterns/    # Pattern recognition visualizations
│   │   └── feature_selection/ # Feature importance visualizations
│   └── reports/         # Generated reports
├── stage_a/             # Stage A: Single/Bagging Forecasting Implementation
│   ├── __init__.py
│   ├── data_acquisition.py     # Data loading and validation
│   ├── preprocessing.py        # Advanced preprocessing methods
│   └── training_strategy.py    # Training strategies
├── stage_b/             # Stage B: Multi-Stage Feature Engineering
│   ├── __init__.py
│   ├── feature_engineering.py  # Technical indicator library
│   ├── pattern_recognition.py  # Pattern detection systems
│   └── feature_selection.py    # Feature importance methods
├── stage_c/             # Stage C: Advanced Prediction Framework (planned)
├── stage_d/             # Stage D: Comprehensive Evaluation (planned)
├── utils/               # Utility functions
│   ├── __init__.py
│   └── visualization.py        # Visualization utilities
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Setup and Installation

1. Clone this repository:
   ```
   git clone [repository_url]
   cd vmethod
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your forex CSV files in the `data/raw/` directory
   - Required files: `EURUSD_1D.csv`, `GBPUSD_1D.csv`, `USDJPY_1D.csv`
   - Required format: CSV with Date, Open, High, Low, Close columns

## Usage

Run the framework with default configuration:

```
python main.py
```

To specify a custom configuration file:

```
python main.py --config path/to/config.yaml
```

To run specific stages:

```
python main.py --stage a  # Run only Stage A
python main.py --stage b  # Run only Stage B
python main.py --stage all  # Run all implemented stages (default)
```

## Stage A Implementation

Stage A focuses on the foundational aspects of the VMETHOD framework:

### Data Acquisition System
- CSV data loading and standardization
- Data validation with quality checks
- Visualization of raw price data

### Advanced Preprocessing Pipeline
- Temporal alignment of multiple currency pairs
- Fractional differentiation for stationarity while preserving memory
- Market regime segmentation (uptrend, downtrend, sideways)
- Wavelet transformation for decomposing price signals

### Training Strategy Implementation
- Chronological train-test splitting
- Single forecast framework (individual currency pairs)
- Bagging forecast framework (combined information)
- Evaluation of different approaches

## Stage B Implementation

Stage B focuses on feature engineering and selection:

### Enhanced Feature Library
- Trend indicators (Moving Averages, MACD, DMI)
- Momentum indicators (RSI, Stochastic, ROC, CCI)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
- Comprehensive visualization of all indicators

### Pattern Recognition System
- Candlestick pattern detection (Doji, Hammer, Engulfing, etc.)
- Technical reversal pattern identification (Head & Shoulders, Double Tops/Bottoms)
- Breakout detection with volume and volatility confirmation
- Visual pattern analysis tools

### Feature Selection Framework
- Random Forest feature importance ranking
- Boruta algorithm for all-relevant feature selection
- SHAP value analysis for interpretable feature importance
- Autoencoder feature selection for non-linear relationships
- Feature fusion system combining multiple methods

## Configuration

The `config/config.yaml` file contains settings for all components of the framework:

- Paths for data and results
- Data acquisition settings
- Preprocessing parameters
- Training strategy options
- Feature engineering configurations
- Pattern recognition thresholds
- Feature selection parameters
- Visualization preferences

Modify this file to customize the behavior of the framework.

## Running Test Cycles

The framework supports multiple train-test cycles as specified in the configuration file:

```yaml
train_test_splits:
  - train_years: ["2020", "2021"]
    test_period: "2022-01/2022-03"
  - train_years: ["2020", "2021", "2022"]
    test_period: "2023-01/2023-03"
  # Additional cycles...
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- pywt (PyWavelets)
- shap
- statsmodels
- mplfinance
- pyyaml

See `requirements.txt` for specific version requirements.

## Future Development

This repository currently implements Stages A and B of the VMETHOD framework. Future development will include:

- Stage C: LSTM, XGBoost, GRU, and TFT models with ensemble techniques
- Stage D: Comprehensive evaluation and trading simulation

## License

[Specify your license here]

## Acknowledgments

[Any acknowledgments or references]