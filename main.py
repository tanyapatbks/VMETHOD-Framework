"""
VMETHOD: A Comprehensive Framework for Forex Market Prediction

Main entry point for executing the VMETHOD framework processes.
This script runs Stage A (data acquisition, preprocessing, and training strategy)
and Stage B (feature engineering, pattern recognition, and feature selection).

Usage:
    python main.py [--config CONFIG_PATH] [--stage STAGE]
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Import VMETHOD components
# Stage A
from stage_a.data_acquisition import DataAcquisitionSystem
from stage_a.preprocessing import AdvancedPreprocessingPipeline
from stage_a.training_strategy import TrainingStrategy

# Stage B
from stage_b.feature_engineering import EnhancedFeatureLibrary
from stage_b.pattern_recognition import PatternRecognitionSystem
from stage_b.feature_selection import FeatureSelectionFramework

# Utilities
from utils.visualization import (
    plot_time_series, 
    plot_correlation_matrix,
    plot_regime_colored_series,
    plot_wavelet_decomposition,
    plot_training_test_split
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vmethod.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VMETHOD")

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def create_directories(config):
    """Create necessary directories based on configuration."""
    directories = [
        config['paths']['raw_data'],
        config['paths']['processed_data'],
        config['paths']['training_data'],
        config['paths']['results'],
        config['paths']['figures'],
        config['paths']['reports'],
        # Add Stage B directories
        os.path.join(config['paths']['results'], 'figures', 'features'),
        os.path.join(config['paths']['results'], 'figures', 'patterns'),
        os.path.join(config['paths']['results'], 'figures', 'feature_selection'),
        'data/features',
        'data/patterns',
        'data/selected_features'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created (if not exists): {directory}")

def run_stage_a(config):
    """
    Execute Stage A of the VMETHOD framework.
    
    Stage A includes:
    1. Data Acquisition
    2. Advanced Preprocessing
    3. Training Strategy Implementation
    """
    logger.info("Starting Stage A: Single/Bagging Forecasting Implementation")
    
    # Step 1: Data Acquisition
    logger.info("Step 1: Data Acquisition")
    acquisition_system = DataAcquisitionSystem(data_dir=config['paths']['raw_data'])
    
    # Load forex data
    currency_pairs = config['data_acquisition']['currency_pairs']
    files = [f"{pair}_1D.csv" for pair in currency_pairs]
    forex_data = acquisition_system.load_csv_data(files)
    
    if not forex_data:
        logger.error("No data loaded. Exiting.")
        return None
    
    # Validate data quality
    validation_report = acquisition_system.validate_data()
    
    # Fix validation issues if needed
    if config['data_acquisition']['interpolate_missing']:
        forex_data = acquisition_system.fix_validation_issues()
    
    # Create data visualizations
    figures_dir = config['paths']['figures']
    for pair_name in forex_data.keys():
        # Time series plot
        acquisition_system.visualize_data(
            pair_name=pair_name, 
            save_path=os.path.join(figures_dir, f"{pair_name}_timeseries.png")
        )
        
        # Candlestick chart
        acquisition_system.create_candlestick_chart(
            pair_name=pair_name,
            save_path=os.path.join(figures_dir, f"{pair_name}_candlestick.png")
        )
    
    # Step 2: Advanced Preprocessing
    logger.info("Step 2: Advanced Preprocessing")
    preprocessing_pipeline = AdvancedPreprocessingPipeline()
    
    # Process all currency pairs
    preprocessed_data = preprocessing_pipeline.process_all_pairs(forex_data)
    
    # Save preprocessed data
    for pair_name, df in preprocessed_data.items():
        output_path = os.path.join(config['paths']['processed_data'], f"{pair_name}_preprocessed.csv")
        df.to_csv(output_path)
        logger.info(f"Preprocessed data saved to {output_path}")
    
    # Visualize preprocessing results
    for pair_name in preprocessed_data.keys():
        preprocessing_pipeline.visualize_preprocessing(
            pair_name=pair_name,
            save_path=os.path.join(figures_dir, f"{pair_name}_preprocessing.png")
        )
        
        # Create regime-colored visualization
        plot_regime_colored_series(
            df=preprocessed_data[pair_name],
            price_col='Close',
            regime_col='Close_regime',
            title=f'{pair_name} Market Regimes',
            save_path=os.path.join(figures_dir, f"{pair_name}_regimes.png")
        )
        
        # Create wavelet decomposition visualization
        plot_wavelet_decomposition(
            df=preprocessed_data[pair_name],
            column='Close',
            title=f'{pair_name} Wavelet Decomposition',
            save_path=os.path.join(figures_dir, f"{pair_name}_wavelet.png")
        )
    
    # Step 3: Training Strategy Implementation
    logger.info("Step 3: Training Strategy Implementation")
    training_strategy = TrainingStrategy(output_dir=config['paths']['training_data'])
    
    # Create train-test splits for each currency pair
    train_years = config['training_strategy']['train_test_splits'][0]['train_years']
    test_period = config['training_strategy']['train_test_splits'][0]['test_period']
    
    splits = training_strategy.create_time_splits(
        data_dict=preprocessed_data,
        train_years=train_years,
        test_period=test_period
    )
    
    # Visualize train-test splits
    for pair_name, split_data in splits.items():
        plot_training_test_split(
            train_df=split_data['train'],
            test_df=split_data['test'],
            column='Close',
            title=f'{pair_name} Train-Test Split',
            save_path=os.path.join(figures_dir, f"{pair_name}_train_test_split.png")
        )
    
    # Create single forecast datasets
    horizon = config['training_strategy']['horizon']
    window = config['training_strategy']['lookback_window']
    
    if config['training_strategy']['approaches']['single_pair']['enabled']:
        single_datasets = training_strategy.create_single_forecast_datasets(
            column='Close',
            horizon=horizon,
            window=window
        )
        logger.info("Created single forecast datasets")
    
    # Create bagging forecast datasets
    if config['training_strategy']['approaches']['bagging']['enabled']:
        bagging_datasets = training_strategy.create_bagging_forecast_datasets(
            column='Close',
            horizon=horizon,
            window=window
        )
        logger.info("Created bagging forecast datasets")
    
    # Visualize all pairs together
    training_strategy.visualize_all_pairs(
        data_dict=forex_data,
        column='Close',
        save_path=os.path.join(figures_dir, "all_pairs_comparison.png")
    )
    
    # Compare single vs bagging approaches
    if (config['training_strategy']['approaches']['single_pair']['enabled'] and 
        config['training_strategy']['approaches']['bagging']['enabled']):
        
        training_strategy.evaluate_approaches(
            save_path=os.path.join(figures_dir, "approach_comparison.png")
        )
    
    logger.info("Stage A completed successfully")
    
    return preprocessed_data

def run_stage_b(config, preprocessed_data):
    """
    Execute Stage B of the VMETHOD framework.
    
    Stage B includes:
    1. Enhanced Feature Library - Technical indicators
    2. Pattern Recognition System
    3. Feature Selection Framework
    
    Args:
        config: Configuration dictionary
        preprocessed_data: Output from Stage A
    """
    if preprocessed_data is None:
        # Try to load preprocessed data from files
        logger.info("No preprocessed data provided, attempting to load from files...")
        preprocessed_data = {}
        currency_pairs = config['data_acquisition']['currency_pairs']
        
        for pair in currency_pairs:
            file_path = os.path.join(config['paths']['processed_data'], f"{pair}_preprocessed.csv")
            if os.path.exists(file_path):
                preprocessed_data[pair] = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded preprocessed data for {pair} from {file_path}")
            else:
                logger.error(f"Preprocessed data file not found for {pair}")
        
        if not preprocessed_data:
            logger.error("No preprocessed data available. Please run Stage A first.")
            return None
    
    logger.info("Starting Stage B: Multi-Stage Feature Engineering")
    
    # Step 1: Enhanced Feature Library
    logger.info("Step 1: Enhanced Feature Library")
    feature_library = EnhancedFeatureLibrary()
    
    # Calculate technical indicators for all pairs
    feature_data = feature_library.calculate_all_features(preprocessed_data)
    
    # Save feature data
    feature_library.save_feature_data(output_dir='data/features/')
    
    # Create feature visualizations
    figures_dir = os.path.join(config['paths']['figures'], 'features')
    for pair_name in feature_data.keys():
        feature_library.create_feature_visualizations(
            pair_name=pair_name,
            output_dir=figures_dir
        )
    
    # Step 2: Pattern Recognition System
    logger.info("Step 2: Pattern Recognition System")
    pattern_system = PatternRecognitionSystem()
    
    # Detect patterns for all pairs
    pattern_data = pattern_system.detect_all_patterns(feature_data)
    
    # Save pattern data
    pattern_system.save_pattern_data(output_dir='data/patterns/')
    
    # Create pattern visualizations
    figures_dir = os.path.join(config['paths']['figures'], 'patterns')
    for pair_name in pattern_data.keys():
        pattern_system.create_pattern_visualizations(
            pair_name=pair_name,
            output_dir=figures_dir
        )
    
    # Step 3: Feature Selection Framework
    logger.info("Step 3: Feature Selection Framework")
    selection_framework = FeatureSelectionFramework()
    
    # Set target column (what we want to predict)
    target_col = 'Close'
    
    # Select features for all pairs
    selected_features_data = selection_framework.select_features(
        data_dict=pattern_data,
        target_col=target_col,
        min_votes=2  # Features must be selected by at least 2 methods
    )
    
    # Save selected features
    selection_framework.save_selected_features(output_dir='data/selected_features/')
    
    # Create feature selection visualizations
    figures_dir = os.path.join(config['paths']['figures'], 'feature_selection')
    selection_framework.create_importance_visualizations(
        target_col=target_col,
        output_dir=figures_dir
    )
    
    logger.info("Stage B completed successfully")
    
    return selected_features_data

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VMETHOD: A Comprehensive Framework for Forex Market Prediction")
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--stage', default='all', choices=['a', 'b', 'all'], help='Stage to run (a, b, or all)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    create_directories(config)
    
    preprocessed_data = None
    selected_features_data = None
    
    # Run requested stages
    if args.stage in ['a', 'all']:
        preprocessed_data = run_stage_a(config)
    
    if args.stage in ['b', 'all']:
        selected_features_data = run_stage_b(config, preprocessed_data)
    
    logger.info("VMETHOD execution completed successfully")

if __name__ == "__main__":
    main()