"""
VMETHOD: A Comprehensive Framework for Forex Market Prediction

Main entry point for executing the VMETHOD framework processes.
This script runs Stage A (data acquisition, preprocessing, and training strategy),
Stage B (feature engineering, pattern recognition, and feature selection), and
Stage C (model development, quantile regression, and ensemble framework).

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
import numpy as np

# Import VMETHOD components
# Stage A
from stage_a.data_acquisition import DataAcquisitionSystem
from stage_a.preprocessing import AdvancedPreprocessingPipeline
from stage_a.training_strategy import TrainingStrategy

# Stage B
from stage_b.feature_engineering import EnhancedFeatureLibrary
from stage_b.pattern_recognition import PatternRecognitionSystem
from stage_b.feature_selection import FeatureSelectionFramework

# Stage C
from stage_c.model_development import ModelDevelopment, LSTMModel, XGBoostModel, GRUModel, TFTModel
from stage_c.quantile_regression import QuantileRegressionSystem
from stage_c.ensemble_framework import EnsembleFramework

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
        'data/selected_features',
        # Add Stage C directories
        'models',
        'models/lstm',
        'models/xgboost',
        'models/gru',
        'models/tft',
        'models/ensemble',
        'models/quantile',
        os.path.join(config['paths']['results'], 'figures', 'models'),
        os.path.join(config['paths']['results'], 'figures', 'quantiles'),
        os.path.join(config['paths']['results'], 'figures', 'ensembles')
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
    
    single_datasets = None
    bagging_datasets = None
    
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
    
    return {
        'preprocessed_data': preprocessed_data,
        'single_datasets': single_datasets,
        'bagging_datasets': bagging_datasets,
        'currency_pairs': currency_pairs
    }

def run_stage_b(config, stage_a_output):
    """
    Execute Stage B of the VMETHOD framework.
    
    Stage B includes:
    1. Enhanced Feature Library - Technical indicators
    2. Pattern Recognition System
    3. Feature Selection Framework
    
    Args:
        config: Configuration dictionary
        stage_a_output: Output from Stage A
    """
    preprocessed_data = stage_a_output.get('preprocessed_data') if stage_a_output else None
    
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
    
    return {
        'feature_data': feature_data,
        'pattern_data': pattern_data,
        'selected_features_data': selected_features_data
    }

def run_stage_c(config, stage_a_output, stage_b_output):
    """
    Execute Stage C of the VMETHOD framework.
    
    Stage C includes:
    1. Model Development - Implementation of LSTM, XGBoost, GRU, and TFT models
    2. Quantile Regression System - Prediction intervals
    3. Ensemble Framework - Methods to combine multiple models
    
    Args:
        config: Configuration dictionary
        stage_a_output: Output from Stage A
        stage_b_output: Output from Stage B
    """
    # Check if we have the necessary inputs
    single_datasets = stage_a_output.get('single_datasets') if stage_a_output else None
    bagging_datasets = stage_a_output.get('bagging_datasets') if stage_a_output else None
    currency_pairs = stage_a_output.get('currency_pairs') if stage_a_output else config['data_acquisition']['currency_pairs']
    
    if single_datasets is None and bagging_datasets is None:
        logger.error("No training datasets available. Please run Stage A first.")
        return None
    
    logger.info("Starting Stage C: Advanced Prediction Framework")
    
    # Load training datasets if not provided
    if single_datasets is None and config['training_strategy']['approaches']['single_pair']['enabled']:
        logger.info("Loading single forecast datasets from files...")
        single_datasets = {}
        
        for pair in currency_pairs:
            pair_datasets = {}
            for set_type in ['train', 'test']:
                X_path = os.path.join(config['paths']['training_data'], f'{pair}_single_{set_type}_X.csv')
                y_path = os.path.join(config['paths']['training_data'], f'{pair}_single_{set_type}_y.csv')
                
                if os.path.exists(X_path) and os.path.exists(y_path):
                    pair_datasets[set_type] = {
                        'X': pd.read_csv(X_path, index_col=0, parse_dates=True),
                        'y': pd.read_csv(y_path, index_col=0, parse_dates=True)
                    }
                else:
                    logger.warning(f"Could not find dataset files for {pair} {set_type}")
            
            if pair_datasets:
                single_datasets[pair] = pair_datasets
    
    if bagging_datasets is None and config['training_strategy']['approaches']['bagging']['enabled']:
        logger.info("Loading bagging forecast datasets from files...")
        bagging_datasets = {}
        
        for set_type in ['train', 'test']:
            X_path = os.path.join(config['paths']['training_data'], f'bagging_{set_type}_X.csv')
            y_path = os.path.join(config['paths']['training_data'], f'bagging_{set_type}_y.csv')
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                bagging_datasets[set_type] = {
                    'X': pd.read_csv(X_path, index_col=0, parse_dates=True),
                    'y': pd.read_csv(y_path, index_col=0, parse_dates=True)
                }
            else:
                logger.warning(f"Could not find bagging dataset files for {set_type}")
    
    # Step 1: Model Development - Training Phase
    logger.info("Step 1.1: Model Training Phase")
    model_development = ModelDevelopment(models_dir='models/')
    
    # Train models using the single-pair approach
    trained_single_models = {}
    if single_datasets and config['training_strategy']['approaches']['single_pair']['enabled']:
        logger.info("Training models using Single-Pair approach")
        
        for pair in currency_pairs:
            logger.info(f"Training models for {pair} (Single approach)")
            
            # Check if data exists for this pair
            if pair not in single_datasets:
                logger.error(f"No data found for {pair}")
                continue
                
            # Get train data
            train_X = single_datasets[pair]['train']['X']
            train_y = single_datasets[pair]['train']['y']
            
            # Create models for this pair
            pair_models = model_development.create_models(approach='single', currency_pair=pair)
            trained_models_for_pair = {}
            
            # Train each model type
            for model_name, model in pair_models.items():
                logger.info(f"Training {model_name} model for {pair}")
                
                try:
                    # Train model with default parameters
                    model.fit(train_X, train_y, verbose=1)
                    
                    # Save trained model
                    model.save()
                    
                    # Store reference to trained model
                    trained_models_for_pair[model_name] = model
                    
                    logger.info(f"Successfully trained {model_name} model for {pair}")
                except Exception as e:
                    logger.error(f"Error training {model_name} model for {pair}: {str(e)}")
            
            # Store trained models for this pair
            if trained_models_for_pair:
                trained_single_models[pair] = trained_models_for_pair
    
    # Train models using the bagging approach
    trained_bagging_models = {}
    if bagging_datasets and config['training_strategy']['approaches']['bagging']['enabled']:
        logger.info("Training models using Bagging approach")
        
        # Get train data
        train_X = bagging_datasets['train']['X']
        train_y = bagging_datasets['train']['y']
        
        # Create models for bagging approach
        bagging_models = model_development.create_models(approach='bagging')
        
        # Train each model type
        for model_name, model in bagging_models.items():
            logger.info(f"Training {model_name} model (Bagging approach)")
            
            try:
                # For bagging, we need to select a specific target column or aggregate them
                # Here we'll take the mean of all targets for simplicity
                combined_y = train_y.mean(axis=1)
                
                # Train model with default parameters
                model.fit(train_X, combined_y, verbose=1)
                
                # Save trained model
                model.save()
                
                # Store reference to trained model
                trained_bagging_models[model_name] = model
                
                logger.info(f"Successfully trained {model_name} model (Bagging approach)")
            except Exception as e:
                logger.error(f"Error training {model_name} model (Bagging approach): {str(e)}")
    
    # Step 1.2: Model Evaluation Phase
    logger.info("Step 1.2: Model Evaluation Phase")
    
    # Evaluate trained single-pair models
    if trained_single_models:
        logger.info("Evaluating Single-Pair models")
        single_evaluation = {}
        
        for pair, models in trained_single_models.items():
            logger.info(f"Evaluating models for {pair}")
            
            # Get test data
            test_X = single_datasets[pair]['test']['X']
            test_y = single_datasets[pair]['test']['y']
            
            pair_evaluation = {}
            
            # Evaluate each trained model
            for model_name, model in models.items():
                metrics = model.evaluate(test_X, test_y)
                pair_evaluation[model_name] = metrics
                
                # Log results
                logger.info(f"Evaluation results for {model_name} on {pair}:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value}")
            
            single_evaluation[pair] = pair_evaluation
        
        # Store single evaluation results
        model_development.evaluation_results['single'] = single_evaluation
    
    # Evaluate trained bagging models
    if trained_bagging_models and bagging_datasets:
        logger.info("Evaluating Bagging models")
        bagging_evaluation = {}
        
        # Get test data
        test_X = bagging_datasets['test']['X']
        test_y = bagging_datasets['test']['y']
        
        # For each currency pair
        for pair in currency_pairs:
            logger.info(f"Evaluating bagging models for {pair}")
            
            # Test data for this pair
            pair_test_y = test_y[pair] if pair in test_y.columns else None
            
            if pair_test_y is None:
                logger.error(f"No test data found for {pair} in bagging dataset")
                continue
                
            pair_evaluation = {}
            
            # Evaluate each trained model
            for model_name, model in trained_bagging_models.items():
                # Generate predictions
                y_pred = model.predict(test_X)
                
                # Convert for metrics calculation
                y_true = pair_test_y.values
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                }
                
                # Calculate directional accuracy
                if len(y_true) > 1:
                    y_diff = np.diff(y_true)
                    y_pred_diff = np.diff(y_pred)
                    correct_direction = (y_diff * y_pred_diff) > 0
                    metrics['directional_accuracy'] = np.mean(correct_direction) * 100
                
                pair_evaluation[model_name] = metrics
                
                # Log results
                logger.info(f"Evaluation results for {model_name} on {pair} (Bagging approach):")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value}")
            
            bagging_evaluation[pair] = pair_evaluation
        
        # Store bagging evaluation results
        model_development.evaluation_results['bagging'] = bagging_evaluation
    
    # Generate performance report
    if model_development.evaluation_results:
        model_development.generate_performance_report(
            save_path=os.path.join(config['paths']['reports'], 'model_performance.csv')
        )
        
        # Visualize performance comparison for different metrics
        for metric in ['rmse', 'mae', 'directional_accuracy']:
            model_development.visualize_performance_comparison(
                metric=metric,
                save_path=os.path.join(config['paths']['results'], 'figures', 'models', f'comparison_{metric}.png')
            )
    
    # Step 2: Quantile Regression System (if any models were successfully trained)
    logger.info("Step 2: Quantile Regression System")
    
    # Find a model that was successfully trained to use with quantile regression
    demo_model = None
    demo_pair = None
    
    # First check single models
    for pair, models in trained_single_models.items():
        if models:  # If there's at least one trained model
            demo_model = next(iter(models.values()))
            demo_pair = pair
            break
    
    # If no single models, check bagging models
    if demo_model is None and trained_bagging_models:
        demo_model = next(iter(trained_bagging_models.values()))
        demo_pair = currency_pairs[0]  # Just use first pair for demonstration
    
    # If we found a trained model, do quantile regression
    if demo_model is not None and demo_pair is not None:
        logger.info(f"Using {demo_model.name} on {demo_pair} for quantile regression demonstration")
        
        # Get the data for this pair
        train_X = single_datasets[demo_pair]['train']['X']
        train_y = single_datasets[demo_pair]['train']['y']
        test_X = single_datasets[demo_pair]['test']['X']
        test_y = single_datasets[demo_pair]['test']['y']
        
        # Initialize quantile system
        quantile_system = QuantileRegressionSystem(model_dir='models/quantile/')
        
        # Train neural network quantile regression
        logger.info(f"Training neural network quantile regression for {demo_pair}")
        quantile_system.train_neural_network(train_X, train_y, epochs=50)
        
        # Plot prediction intervals
        quantile_system.plot_prediction_intervals(
            test_X, test_y,
            save_path=os.path.join(config['paths']['results'], 'figures', 'quantiles', f'{demo_pair}_nn_intervals.png')
        )
        
        # Plot calibration
        quantile_system.plot_calibration(
            test_X, test_y,
            save_path=os.path.join(config['paths']['results'], 'figures', 'quantiles', f'{demo_pair}_nn_calibration.png')
        )
        
        # Train XGBoost quantile regression
        logger.info(f"Training XGBoost quantile regression for {demo_pair}")
        quantile_system.train_xgboost(train_X, train_y)
        
        # Plot prediction intervals
        quantile_system.plot_prediction_intervals(
            test_X, test_y, model_type='xgboost',
            save_path=os.path.join(config['paths']['results'], 'figures', 'quantiles', f'{demo_pair}_xgb_intervals.png')
        )
        
        # Plot calibration
        quantile_system.plot_calibration(
            test_X, test_y, model_type='xgboost',
            save_path=os.path.join(config['paths']['results'], 'figures', 'quantiles', f'{demo_pair}_xgb_calibration.png')
        )
        
        # Save quantile regression system
        quantile_system.save()
    else:
        logger.warning("No trained models available for quantile regression demonstration")
    
    # Step 3: Ensemble Framework (if we have multiple trained models for a pair)
    logger.info("Step 3: Ensemble Framework")
    
    # Find a pair with multiple trained models for ensemble demonstration
    ensemble_pair = None
    ensemble_models = {}
    
    # Check if any pair has multiple trained models
    for pair, models in trained_single_models.items():
        if len(models) >= 2:  # Need at least 2 models for ensemble
            ensemble_pair = pair
            ensemble_models = models
            break
    
    # If we found multiple trained models for a pair, do ensemble
    if ensemble_pair is not None and len(ensemble_models) >= 2:
        logger.info(f"Creating ensemble for {ensemble_pair} with {len(ensemble_models)} models")
        
        # Get the data for this pair
        train_X = single_datasets[ensemble_pair]['train']['X']
        train_y = single_datasets[ensemble_pair]['train']['y']
        test_X = single_datasets[ensemble_pair]['test']['X']
        test_y = single_datasets[ensemble_pair]['test']['y']
        
        # Initialize ensemble framework
        ensemble_framework = EnsembleFramework(model_dir='models/ensemble/')
        
        # Add trained models to ensemble
        for model_name, model in ensemble_models.items():
            ensemble_framework.add_model(model)
        
        # Train stacking ensemble
        logger.info(f"Training stacking ensemble for {ensemble_pair}")
        ensemble_framework.train_stacking_ensemble(train_X, train_y)
        
        # Plot model weights
        ensemble_framework.plot_model_weights(
            save_path=os.path.join(config['paths']['results'], 'figures', 'ensembles', f'{ensemble_pair}_weights.png')
        )
        
        # Analyze ensemble diversity
        ensemble_framework.analyze_ensemble_diversity(
            test_X, test_y,
            save_path=os.path.join(config['paths']['results'], 'figures', 'ensembles', f'{ensemble_pair}_diversity.png')
        )
        
        # Save ensemble framework
        ensemble_framework.save()
        
        # Check if we have regime data for regime-specific ensemble
        if 'Close_regime' in train_X.columns:
            logger.info(f"Training regime-specific ensemble for {ensemble_pair}")
            
            # Create a new ensemble for regime-specific approach
            regime_ensemble = EnsembleFramework(model_dir='models/ensemble/regime/')
            
            # Add trained models to regime ensemble
            for model_name, model in ensemble_models.items():
                regime_ensemble.add_model(model)
            
            # Train regime-specific ensemble
            regime_ensemble.train_regime_specific(train_X, train_y, 'Close_regime', [-1, 0, 1])
            
            # Plot model weights for different regimes
            for regime in [-1, 0, 1]:
                regime_ensemble.plot_model_weights(
                    regime=regime,
                    save_path=os.path.join(config['paths']['results'], 'figures', 'ensembles', 
                                         f'{ensemble_pair}_regime{regime}_weights.png')
                )
            
            # Save regime-specific ensemble
            regime_ensemble.save()
    else:
        logger.warning("Not enough trained models available for ensemble demonstration")
    
    logger.info("Stage C completed successfully")
    
    return {
        'model_development': model_development,
        'trained_single_models': trained_single_models,
        'trained_bagging_models': trained_bagging_models
    }

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VMETHOD: A Comprehensive Framework for Forex Market Prediction")
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--stage', default='all', choices=['a', 'b', 'c', 'all'], help='Stage to run (a, b, c, or all)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    create_directories(config)
    
    stage_a_output = None
    stage_b_output = None
    stage_c_output = None
    
    # Run requested stages
    if args.stage in ['a', 'all']:
        stage_a_output = run_stage_a(config)
    
    if args.stage in ['b', 'all']:
        stage_b_output = run_stage_b(config, stage_a_output)
    
    if args.stage in ['c', 'all']:
        stage_c_output = run_stage_c(config, stage_a_output, stage_b_output)
    
    logger.info("VMETHOD execution completed successfully")

if __name__ == "__main__":
    main()