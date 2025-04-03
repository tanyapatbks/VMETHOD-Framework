"""
Quantile Regression System for the VMETHOD framework.

This module provides prediction intervals rather than single-point forecasts,
quantifying uncertainty in market predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import os
import logging
import time
from datetime import datetime
import joblib
import json

# ML libraries
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# For XGBoost quantile regression
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QuantileRegression')


def quantile_loss(q, y_true, y_pred):
    """
    Custom quantile loss function for TensorFlow.
    
    Args:
        q: Quantile to compute (0.1 for 10th percentile, 0.9 for 90th percentile)
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Quantile loss
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))


class QuantileRegressionSystem:
    """
    Quantile Regression System for providing prediction intervals.
    
    This class provides methods to train models that predict multiple quantiles,
    allowing for the quantification of uncertainty in predictions.
    """
    
    def __init__(self, model_dir: str = 'models/quantile/', 
                quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        Initialize the Quantile Regression System.
        
        Args:
            model_dir: Directory to save model artifacts
            quantiles: List of quantiles to predict (0.5 is the median)
        """
        self.model_dir = model_dir
        self.quantiles = quantiles
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.numeric_columns = None
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def _build_neural_network(self, input_dim: int) -> Dict[float, tf.keras.Model]:
        """
        Build neural network models for each quantile.
        
        Args:
            input_dim: Input dimension (number of features)
            
        Returns:
            Dictionary of neural network models for each quantile
        """
        models = {}
        
        for q in self.quantiles:
            # Create a model for this quantile
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Create a custom loss function with the specific quantile
            q_loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred)
            q_loss.__name__ = f'quantile_loss_{q}'
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss=q_loss)
            
            models[q] = model
            
        return models
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.DataFrame, 
                            validation_split: float = 0.2, epochs: int = 100, 
                            batch_size: int = 32, patience: int = 20, 
                            verbose: int = 1) -> None:
        """
        Train neural network models for quantile regression.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level for training
        """
        # Select only numeric columns for model training
        X_numeric = X.select_dtypes(include=['number'])
        logger.info(f"Selected {X_numeric.shape[1]} numeric features from {X.shape[1]} total features")
        
        # Store numeric column names for future use
        self.numeric_columns = X_numeric.columns.tolist()
        
        # Preprocess data
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_numeric)
        
        # Scale target
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Save scalers
        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y
        
        # Build models for each quantile
        self.models['neural_network'] = self._build_neural_network(X_scaled.shape[1])
        
        # Train models for each quantile
        for q, model in self.models['neural_network'].items():
            logger.info(f"Training neural network for quantile {q}")
            
            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            ]
            
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.model_dir, f"nn_q{q}_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Add model checkpoint callback
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
                    save_best_only=True,
                    monitor='val_loss'
                )
            )
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_scaled, y_scaled,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            training_time = time.time() - start_time
            
            # Log training results
            logger.info(f"Neural network for quantile {q} trained for {len(history.history['loss'])} epochs "
                      f"in {training_time:.2f} seconds")
            
        self.is_trained = True
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.DataFrame,
                     n_estimators: int = 100, learning_rate: float = 0.1,
                     max_depth: int = 5, subsample: float = 0.8,
                     early_stopping_rounds: int = 20, verbose: int = 1) -> None:
        """
        Train XGBoost models for quantile regression.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio
            early_stopping_rounds: Patience for early stopping
            verbose: Verbosity level for training
        """
        # Select only numeric columns for model training
        X_numeric = X.select_dtypes(include=['number'])
        logger.info(f"Selected {X_numeric.shape[1]} numeric features from {X.shape[1]} total features")
        
        # Store numeric column names for future use
        self.numeric_columns = X_numeric.columns.tolist()
        
        # Preprocess data
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_numeric)
        
        # Scale target
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Save scalers
        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y
        
        # Initialize dictionary to store XGBoost models
        self.models['xgboost'] = {}
        
        # Split data for validation
        n = len(X_scaled)
        split_idx = int(n * 0.8)
        
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train models for each quantile
        for q in self.quantiles:
            logger.info(f"Training XGBoost for quantile {q}")
            
            # Set parameters for quantile regression
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': q,  # Specify the quantile
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'seed': 42
            }
            
            # Define evaluation list
            eval_list = [(dtrain, 'train'), (dval, 'validation')]
            
            # Train model
            start_time = time.time()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=eval_list,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose
            )
            
            training_time = time.time() - start_time
            
            # Save model
            self.models['xgboost'][q] = model
            
            # Log training results
            logger.info(f"XGBoost for quantile {q} trained in {training_time:.2f} seconds")
            
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame, model_type: str = 'neural_network') -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions.
        
        Args:
            X: Feature DataFrame
            model_type: Type of model to use ('neural_network' or 'xgboost')
            
        Returns:
            Dictionary of predictions for each quantile
        """
        if not self.is_trained or model_type not in self.models:
            raise ValueError(f"Models of type {model_type} not trained. Call train_{model_type} first.")
            
        # Select only numeric columns
        if self.numeric_columns:
            X_numeric = X[self.numeric_columns]
        else:
            X_numeric = X.select_dtypes(include=['number'])
            logger.warning("Using all numeric columns as numeric_columns is not set")
        
        # Preprocess data
        X_scaled = self.scalers['X'].transform(X_numeric)
        
        # Initialize dictionary to store predictions
        predictions = {}
        
        if model_type == 'neural_network':
            # Generate predictions for each quantile
            for q, model in self.models['neural_network'].items():
                pred_scaled = model.predict(X_scaled)
                
                # Inverse transform predictions
                pred = self.scalers['y'].inverse_transform(pred_scaled).flatten()
                
                predictions[q] = pred
                
        elif model_type == 'xgboost':
            # Convert to DMatrix for XGBoost
            dtest = xgb.DMatrix(X_scaled)
            
            # Generate predictions for each quantile
            for q, model in self.models['xgboost'].items():
                pred_scaled = model.predict(dtest)
                
                # Inverse transform predictions
                pred = self.scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                predictions[q] = pred
                
        return predictions
    
    def calculate_prediction_intervals(self, X: pd.DataFrame, model_type: str = 'neural_network',
                                     low_quantile: float = 0.1, high_quantile: float = 0.9) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals using specified quantiles.
        
        Args:
            X: Feature DataFrame
            model_type: Type of model to use ('neural_network' or 'xgboost')
            low_quantile: Lower bound quantile
            high_quantile: Upper bound quantile
            
        Returns:
            Dictionary with predictions and bounds
        """
        if low_quantile not in self.quantiles or high_quantile not in self.quantiles:
            raise ValueError(f"Specified quantiles must be in {self.quantiles}")
            
        # Get quantile predictions
        quantile_preds = self.predict(X, model_type)
        
        # Extract median (or closest to 0.5), lower, and upper bounds
        if 0.5 in quantile_preds:
            median_pred = quantile_preds[0.5]
        else:
            # Find closest quantile to 0.5
            closest_to_median = min(self.quantiles, key=lambda x: abs(x - 0.5))
            median_pred = quantile_preds[closest_to_median]
            
        low_bound = quantile_preds[low_quantile]
        high_bound = quantile_preds[high_quantile]
        
        return {
            'median': median_pred,
            'lower_bound': low_bound,
            'upper_bound': high_bound,
            'interval_width': high_bound - low_bound
        }
    
    def evaluate_calibration(self, X: pd.DataFrame, y: pd.DataFrame,
                           model_type: str = 'neural_network') -> Dict[float, float]:
        """
        Evaluate the calibration of quantile predictions.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            model_type: Type of model to use ('neural_network' or 'xgboost')
            
        Returns:
            Dictionary of calibration scores for each quantile
        """
        # Get quantile predictions
        quantile_preds = self.predict(X, model_type)
        
        # Convert y to numpy array
        y_true = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Calculate calibration for each quantile
        calibration = {}
        
        for q, pred in quantile_preds.items():
            # For a perfectly calibrated quantile estimator, the true values should fall below
            # the predicted quantile roughly q% of the time
            hits = (y_true <= pred).mean()
            
            # Calculate calibration error (how far from expected quantile)
            error = abs(hits - q)
            
            calibration[q] = {
                'expected': q,
                'observed': hits,
                'error': error
            }
            
        return calibration
    
    def plot_prediction_intervals(self, X: pd.DataFrame, y: pd.DataFrame,
                                model_type: str = 'neural_network',
                                low_quantile: float = 0.1, high_quantile: float = 0.9,
                                save_path: Optional[str] = None) -> None:
        """
        Plot prediction intervals against actual values.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            model_type: Type of model to use ('neural_network' or 'xgboost')
            low_quantile: Lower bound quantile
            high_quantile: Upper bound quantile
            save_path: Path to save the plot
        """
        # Calculate prediction intervals
        intervals = self.calculate_prediction_intervals(X, model_type, low_quantile, high_quantile)
        
        # Convert y to numpy array
        y_true = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot median prediction
        plt.plot(intervals['median'], label='Median Prediction', color='blue')
        
        # Plot actual values
        plt.plot(y_true, label='Actual', color='black', linestyle='--')
        
        # Plot prediction intervals
        plt.fill_between(
            range(len(intervals['median'])),
            intervals['lower_bound'],
            intervals['upper_bound'],
            alpha=0.2,
            color='blue',
            label=f"{int(low_quantile*100)}%-{int(high_quantile*100)}% Interval"
        )
        
        # Calculate percentage of points within the interval
        coverage = np.mean((y_true >= intervals['lower_bound']) & (y_true <= intervals['upper_bound'])) * 100
        expected_coverage = (high_quantile - low_quantile) * 100
        
        plt.title(f"Prediction Intervals ({model_type.capitalize()})\n"
                f"Expected coverage: {expected_coverage:.1f}%, Actual coverage: {coverage:.1f}%")
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction intervals plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_calibration(self, X: pd.DataFrame, y: pd.DataFrame,
                       model_type: str = 'neural_network',
                       save_path: Optional[str] = None) -> None:
        """
        Plot calibration of quantile predictions.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            model_type: Type of model to use ('neural_network' or 'xgboost')
            save_path: Path to save the plot
        """
        # Get calibration scores
        calibration = self.evaluate_calibration(X, y, model_type)
        
        # Extract data for plotting
        quantiles = []
        observed = []
        expected = []
        
        for q, scores in calibration.items():
            quantiles.append(q)
            observed.append(scores['observed'])
            expected.append(scores['expected'])
            
        # Sort by quantile
        idx = np.argsort(quantiles)
        quantiles = np.array(quantiles)[idx]
        observed = np.array(observed)[idx]
        expected = np.array(expected)[idx]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot expected calibration line
        plt.plot(quantiles, expected, label='Expected (Perfect Calibration)', color='black', linestyle='--')
        
        # Plot observed calibration
        plt.plot(quantiles, observed, label='Observed', color='blue', marker='o')
        
        # Fill the area between to highlight deviation
        plt.fill_between(quantiles, expected, observed, alpha=0.2, color='red')
        
        # Add reference lines
        plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0.5, color='gray', linestyle='-', alpha=0.3)
        
        plt.title(f"Quantile Calibration ({model_type.capitalize()})")
        plt.xlabel('Expected Quantile')
        plt.ylabel('Observed Quantile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with calibration error
        mean_abs_error = np.mean([scores['error'] for scores in calibration.values()])
        plt.text(0.05, 0.95, f'Mean Abs. Calibration Error: {mean_abs_error:.4f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")
        else:
            plt.show()
    
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the quantile regression system.
        
        Args:
            save_path: Path to save the system (if None, generate based on timestamp)
            
        Returns:
            Path where system was saved
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.model_dir, f"quantile_system_{timestamp}")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'quantiles': self.quantiles,
            'is_trained': self.is_trained,
            'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'numeric_columns': self.numeric_columns
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save scalers
        if self.scalers:
            for scaler_name, scaler in self.scalers.items():
                joblib.dump(scaler, os.path.join(save_path, f'scaler_{scaler_name}.joblib'))
            
        # Save neural network models if available
        if 'neural_network' in self.models:
            nn_dir = os.path.join(save_path, 'neural_network')
            os.makedirs(nn_dir, exist_ok=True)
            
            for q, model in self.models['neural_network'].items():
                model.save(os.path.join(nn_dir, f'model_q{q}.h5'))
        
        # Save XGBoost models if available
        if 'xgboost' in self.models:
            xgb_dir = os.path.join(save_path, 'xgboost')
            os.makedirs(xgb_dir, exist_ok=True)
            
            for q, model in self.models['xgboost'].items():
                model.save_model(os.path.join(xgb_dir, f'model_q{q}.json'))
        
        logger.info(f"Quantile regression system saved to {save_path}")
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the quantile regression system.
        
        Args:
            load_path: Path to load the system from
        """
        # Load metadata
        with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.quantiles = metadata['quantiles']
        self.is_trained = metadata['is_trained']
        self.numeric_columns = metadata.get('numeric_columns', None)
        
        # Load scalers
        scaler_X_path = os.path.join(load_path, 'scaler_X.joblib')
        scaler_y_path = os.path.join(load_path, 'scaler_y.joblib')
        
        if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
            self.scalers['X'] = joblib.load(scaler_X_path)
            self.scalers['y'] = joblib.load(scaler_y_path)
        
        # Load neural network models if available
        nn_dir = os.path.join(load_path, 'neural_network')
        if os.path.exists(nn_dir):
            self.models['neural_network'] = {}
            
            for q in self.quantiles:
                model_path = os.path.join(nn_dir, f'model_q{q}.h5')
                if os.path.exists(model_path):
                    # Load model with custom loss function
                    q_loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred)
                    q_loss.__name__ = f'quantile_loss_{q}'
                    
                    self.models['neural_network'][q] = load_model(
                        model_path,
                        custom_objects={f'quantile_loss_{q}': q_loss}
                    )
        
        # Load XGBoost models if available
        xgb_dir = os.path.join(load_path, 'xgboost')
        if os.path.exists(xgb_dir):
            self.models['xgboost'] = {}
            
            for q in self.quantiles:
                model_path = os.path.join(xgb_dir, f'model_q{q}.json')
                if os.path.exists(model_path):
                    model = xgb.Booster()
                    model.load_model(model_path)
                    self.models['xgboost'][q] = model
        
        logger.info(f"Quantile regression system loaded from {load_path}")