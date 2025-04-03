"""
Model Development for the VMETHOD framework.

This module implements the base models used for forex prediction:
- LSTM (Long Short-Term Memory)
- XGBoost
- GRU (Gated Recurrent Unit)
- TFT (Temporal Fusion Transformer)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import os
import logging
import time
import joblib
import json
from datetime import datetime

# ML/DL libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, Attention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelDevelopment')

class BaseModel:
    """Base class for all prediction models."""
    
    def __init__(self, name: str, model_dir: str = 'models/'):
        """
        Initialize base model.
        
        Args:
            name: Model name
            model_dir: Directory to save model artifacts
        """
        self.name = name
        self.model_dir = model_dir
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_trained = False
        self.training_history = None
        self.feature_importance = None
        self.numeric_columns = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def preprocess(self, X: pd.DataFrame, y: pd.DataFrame = None, fit: bool = True) -> Tuple:
        """
        Preprocess the input data.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series (optional)
            fit: Whether to fit the scalers on this data
            
        Returns:
            Tuple of preprocessed X and optionally y
        """
        # Select only numeric columns for model training
        X_numeric = X.select_dtypes(include=['number'])
        logger.info(f"Selected {X_numeric.shape[1]} numeric features from {X.shape[1]} total features")
        
        # Store numeric column names for future use
        if fit:
            self.numeric_columns = X_numeric.columns.tolist()
        elif self.numeric_columns:
            # Use stored numeric columns for consistency
            X_numeric = X[self.numeric_columns]
        
        if fit:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X_numeric)
            
            if y is not None:
                self.scaler_y = StandardScaler()
                y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
                return X_scaled, y_scaled
        else:
            if self.scaler_X is None:
                raise ValueError("Scalers not initialized. Call preprocess with fit=True first.")
                
            X_scaled = self.scaler_X.transform(X_numeric)
            
            if y is not None and self.scaler_y is not None:
                y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
                return X_scaled, y_scaled
        
        return X_scaled
    
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled y values.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        if self.scaler_y is None:
            raise ValueError("y scaler not initialized")
            
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the model and associated artifacts.
        
        Args:
            save_path: Path to save model (if None, generate based on name)
            
        Returns:
            Path where model was saved
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.model_dir, f"{self.name}_{timestamp}")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'is_trained': self.is_trained,
            'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'numeric_columns': self.numeric_columns
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save scalers
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, os.path.join(save_path, 'scaler_X.joblib'))
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, os.path.join(save_path, 'scaler_y.joblib'))
            
        # Save feature importance if available
        if self.feature_importance is not None:
            pd.Series(self.feature_importance).to_csv(os.path.join(save_path, 'feature_importance.csv'))
            
        logger.info(f"Model {self.name} metadata and scalers saved to {save_path}")
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the model and associated artifacts.
        
        Args:
            load_path: Path to load model from
        """
        # Load metadata
        with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.name = metadata['name']
        self.is_trained = metadata['is_trained']
        self.numeric_columns = metadata.get('numeric_columns', None)
        
        # Load scalers
        scaler_X_path = os.path.join(load_path, 'scaler_X.joblib')
        if os.path.exists(scaler_X_path):
            self.scaler_X = joblib.load(scaler_X_path)
            
        scaler_y_path = os.path.join(load_path, 'scaler_y.joblib')
        if os.path.exists(scaler_y_path):
            self.scaler_y = joblib.load(scaler_y_path)
            
        # Load feature importance if available
        feature_importance_path = os.path.join(load_path, 'feature_importance.csv')
        if os.path.exists(feature_importance_path):
            self.feature_importance = pd.read_csv(feature_importance_path, index_col=0, squeeze=True).to_dict()
            
        logger.info(f"Model {self.name} metadata and scalers loaded from {load_path}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit first.")
            
        y_pred = self.predict(X)
        
        # Convert to numpy arrays for metric calculation
        y_true = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Calculate directional accuracy (correct prediction of up/down movement)
        if len(y_true) > 1:
            y_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            correct_direction = (y_diff * y_pred_diff) > 0
            metrics['directional_accuracy'] = np.mean(correct_direction) * 100
        
        return metrics
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.training_history is None:
            logger.warning("No training history available")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Train Loss')
        if 'val_loss' in self.training_history:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot other metrics if available
        metrics = [key for key in self.training_history.keys() 
                  if key not in ['loss', 'val_loss', 'lr']]
        
        if metrics:
            plt.subplot(1, 2, 2)
            for metric in metrics:
                plt.plot(self.training_history[metric], label=metric)
            plt.title('Metrics History')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot model predictions vs actual values.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            save_path: Path to save the plot
        """
        if not self.is_trained:
            logger.warning("Model not trained. Cannot generate predictions.")
            return
            
        # Generate predictions
        y_pred = self.predict(X)
        
        # Convert to numpy arrays
        y_true = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        plt.title(f'{self.name} - Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot prediction error
        plt.subplot(2, 1, 2)
        error = y_true - y_pred
        plt.plot(error, color='green')
        plt.title('Prediction Error')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_names: List[str] = None, top_n: int = 20, 
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
            
        # Convert to Series for easier handling
        if isinstance(self.feature_importance, dict):
            importance = pd.Series(self.feature_importance)
        else:
            importance = pd.Series(self.feature_importance, index=feature_names)
            
        # Sort and get top features
        importance = importance.sort_values(ascending=False)
        if top_n and len(importance) > top_n:
            importance = importance.head(top_n)
            
        # Create plot
        plt.figure(figsize=(10, max(8, len(importance) * 0.3)))
        plt.barh(importance.index, importance.values)
        plt.title(f'{self.name} - Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()


class LSTMModel(BaseModel):
    """LSTM (Long Short-Term Memory) implementation for time series forecasting."""
    
    def __init__(self, name: str = "LSTM", model_dir: str = 'models/', 
                units: int = 64, dropout: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize LSTM model.
        
        Args:
            name: Model name
            model_dir: Directory to save model artifacts
            units: Number of LSTM units
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        super().__init__(name, model_dir)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input features (samples, features)
        """
        
        # Create sequential model
        self.model = Sequential([
            
            # LSTM layers
            LSTM(self.units, return_sequences=True, input_shape=(1, input_shape[1])),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            
            # Output layer
            Dense(1)
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"LSTM model built with {self.units} units and {self.dropout} dropout")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, validation_split: float = 0.2,
          epochs: int = 100, batch_size: int = 32, patience: int = 20,
          verbose: int = 1) -> None:
        """
        Train the LSTM model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level for training
        """
        # Preprocess data
        X_scaled, y_scaled = self.preprocess(X, y, fit=True)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_scaled.shape)
            
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
        ]
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.model_dir, f"{self.name}_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add model checkpoint callback
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        )
        
        # Reshape X for LSTM [samples, timesteps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            X_reshaped, y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = history.history
        self.is_trained = True
        
        # Log training results
        logger.info(f"LSTM model trained for {len(history.history['loss'])} epochs in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained LSTM model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit first.")
            
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Generate predictions
        y_pred_scaled = self.model.predict(X_reshaped)
        
        # Inverse transform predictions
        y_pred = self.inverse_transform_y(y_pred_scaled)
        
        return y_pred
        
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the LSTM model and associated artifacts.
        
        Args:
            save_path: Path to save model (if None, generate based on name)
            
        Returns:
            Path where model was saved
        """
        save_path = super().save(save_path)
        
        # Save Keras model
        if self.model is not None:
            self.model.save(os.path.join(save_path, 'keras_model.h5'))
            
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the LSTM model and associated artifacts.
        
        Args:
            load_path: Path to load model from
        """
        super().load(load_path)
        
        # Load Keras model
        model_path = os.path.join(load_path, 'keras_model.h5')
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            logger.warning(f"No Keras model found at {model_path}")


class XGBoostModel(BaseModel):
    """XGBoost implementation for time series forecasting."""
    
    def __init__(self, name: str = "XGBoost", model_dir: str = 'models/',
                n_estimators: int = 100, learning_rate: float = 0.1,
                max_depth: int = 5, subsample: float = 0.8):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name
            model_dir: Directory to save model artifacts
            n_estimators: Number of boosting rounds
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio
        """
        super().__init__(name, model_dir)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, validation_split: float = 0.2,
          early_stopping_rounds: int = 20, verbose: int = 1) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            validation_split: Fraction of data to use for validation
            early_stopping_rounds: Patience for early stopping
            verbose: Verbosity level for training
        """
        # Preprocess data
        X_scaled, y_scaled = self.preprocess(X, y, fit=True)
        
        # Split data for validation
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Convert to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'seed': 42
        }
        
        # Define evaluation list
        eval_list = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model
        start_time = time.time()
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_list,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose
        )
        
        training_time = time.time() - start_time
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Create a simplified training history dictionary
        # Store evaluation results manually
        evals_result = {}  # Create empty dict to store results
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_list,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose,
            evals_result=evals_result  # Pass as parameter to capture results
        )

        # Now use the evals_result dict instead of model.eval_result
        self.training_history = {
            'loss': evals_result['train']['rmse'],
            'val_loss': evals_result['validation']['rmse']
        }
        
        self.is_trained = True
        
        # Log training results
        logger.info(f"XGBoost model trained in {training_time:.2f} seconds")
        logger.info(f"Best iteration: {self.model.best_iteration}")
        logger.info(f"Best training RMSE: {evals_result['train']['rmse'][self.model.best_iteration]:.4f}")
        logger.info(f"Best validation RMSE: {evals_result['validation']['rmse'][self.model.best_iteration]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained XGBoost model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit first.")
            
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X_scaled)
        
        # Generate predictions
        y_pred_scaled = self.model.predict(dtest)
        
        # Inverse transform predictions
        y_pred = self.inverse_transform_y(y_pred_scaled)
        
        return y_pred
        
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the XGBoost model and associated artifacts.
        
        Args:
            save_path: Path to save model (if None, generate based on name)
            
        Returns:
            Path where model was saved
        """
        save_path = super().save(save_path)
        
        # Save XGBoost model
        if self.model is not None:
            self.model.save_model(os.path.join(save_path, 'xgboost_model.json'))
            
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the XGBoost model and associated artifacts.
        
        Args:
            load_path: Path to load model from
        """
        super().load(load_path)
        
        # Load XGBoost model
        model_path = os.path.join(load_path, 'xgboost_model.json')
        if os.path.exists(model_path):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            logger.warning(f"No XGBoost model found at {model_path}")


class GRUModel(BaseModel):
    """GRU (Gated Recurrent Unit) implementation for time series forecasting."""
    
    def __init__(self, name: str = "GRU", model_dir: str = 'models/', 
                units: int = 64, dropout: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize GRU model.
        
        Args:
            name: Model name
            model_dir: Directory to save model artifacts
            units: Number of GRU units
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        super().__init__(name, model_dir)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input features (samples, features)
        """
        # Reshape input shape for GRU [samples, timesteps, features]
        # For non-sequence inputs, we use timestep of 1
        gru_input_shape = (1, input_shape[1])
        
        # Remove the reshape layer
        self.model = Sequential([
            # Use input_shape=(1, features) directly for GRU
            tf.keras.layers.Bidirectional(GRU(self.units, return_sequences=True, 
                                            input_shape=(1, input_shape[1]))),
            Dropout(self.dropout),
            tf.keras.layers.Bidirectional(GRU(self.units // 2)),
            Dropout(self.dropout),
            
            # Output layer
            Dense(1)
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"GRU model built with {self.units} units and {self.dropout} dropout")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, validation_split: float = 0.2,
          epochs: int = 100, batch_size: int = 32, patience: int = 20,
          verbose: int = 1) -> None:
        """
        Train the GRU model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level for training
        """
        # Preprocess data
        X_scaled, y_scaled = self.preprocess(X, y, fit=True)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_scaled.shape)
            
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
        ]
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.model_dir, f"{self.name}_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add model checkpoint callback
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        )
        
        # Reshape X for GRU [samples, timesteps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            X_reshaped, y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = history.history
        self.is_trained = True
        
        # Log training results
        logger.info(f"GRU model trained for {len(history.history['loss'])} epochs in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained GRU model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit first.")
            
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Reshape for GRU [samples, timesteps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Generate predictions
        y_pred_scaled = self.model.predict(X_reshaped)
        
        # Inverse transform predictions
        y_pred = self.inverse_transform_y(y_pred_scaled)
        
        return y_pred
        
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the GRU model and associated artifacts.
        
        Args:
            save_path: Path to save model (if None, generate based on name)
            
        Returns:
            Path where model was saved
        """
        save_path = super().save(save_path)
        
        # Save Keras model
        if self.model is not None:
            self.model.save(os.path.join(save_path, 'keras_model.h5'))
            
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the GRU model and associated artifacts.
        
        Args:
            load_path: Path to load model from
        """
        super().load(load_path)
        
        # Load Keras model
        model_path = os.path.join(load_path, 'keras_model.h5')
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            logger.warning(f"No Keras model found at {model_path}")


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.query_layer = Dense(units)
        self.key_layer = Dense(units)
        self.value_layer = Dense(units)
        
    def call(self, inputs):
        """
        Apply attention mechanism.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Context vector and attention weights
        """
        # Project inputs
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        
        # Calculate dot product attention
        score = tf.matmul(query, key, transpose_b=True)
        
        # Scale score
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        score = score / tf.math.sqrt(dk)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Apply attention weights to values
        context = tf.matmul(attention_weights, value)
        
        return context, attention_weights


class TFTModel(BaseModel):
    """Temporal Fusion Transformer (TFT) implementation for time series forecasting."""
    
    def __init__(self, name: str = "TFT", model_dir: str = 'models/', 
                hidden_units: int = 64, attention_heads: int = 4,
                dropout: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize TFT model.
        
        Args:
            name: Model name
            model_dir: Directory to save model artifacts
            hidden_units: Number of hidden units
            attention_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate for Adam optimizer
        """
        super().__init__(name, model_dir)
        self.hidden_units = hidden_units
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build TFT model architecture.
        
        Args:
            input_shape: Shape of input features (samples, features)
        """
        # Define input
        inputs = Input(shape=(input_shape[1],))
        
        # Reshape for temporal processing [samples, timesteps, features]
        # We'll treat each feature as a separate time point for attention
        seq_length = input_shape[1]
        x = tf.keras.layers.Reshape((seq_length, 1))(inputs)
        
        # Initial processing with dense layers
        x = Dense(self.hidden_units, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        
        # Layer normalization
        x = LayerNormalization()(x)
        
        # Multi-head attention
        attention_output = []
        for _ in range(self.attention_heads):
            attention = TemporalAttention(self.hidden_units)
            attn_out, _ = attention(x)
            attention_output.append(attn_out)
            
        # Combine attention heads
        x = tf.concat(attention_output, axis=-1)
        x = Dense(self.hidden_units)(x)
        x = Dropout(self.dropout)(x)
        
        # Layer normalization
        x = LayerNormalization()(x)
        
        # Global average pooling over the sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"TFT model built with {self.hidden_units} units and {self.attention_heads} attention heads")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, validation_split: float = 0.2,
          epochs: int = 100, batch_size: int = 32, patience: int = 20,
          verbose: int = 1) -> None:
        """
        Train the TFT model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Verbosity level for training
        """
        # Preprocess data
        X_scaled, y_scaled = self.preprocess(X, y, fit=True)
        
        # Build model if not already built
        if self.model is None:
            self._build_model(X_scaled.shape)
            
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6)
        ]
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.model_dir, f"{self.name}_checkpoints")
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
        history = self.model.fit(
            X_scaled, y_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history = history.history
        self.is_trained = True
        
        # Log training results
        logger.info(f"TFT model trained for {len(history.history['loss'])} epochs in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        if 'val_loss' in history.history:
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained TFT model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call fit first.")
            
        # Preprocess data
        X_scaled = self.preprocess(X, fit=False)
        
        # Generate predictions
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        y_pred = self.inverse_transform_y(y_pred_scaled)
        
        return y_pred
        
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the TFT model and associated artifacts.
        
        Args:
            save_path: Path to save model (if None, generate based on name)
            
        Returns:
            Path where model was saved
        """
        save_path = super().save(save_path)
        
        # Save Keras model
        if self.model is not None:
            self.model.save(os.path.join(save_path, 'keras_model.h5'))
            
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the TFT model and associated artifacts.
        
        Args:
            load_path: Path to load model from
        """
        super().load(load_path)
        
        # Load Keras model
        model_path = os.path.join(load_path, 'keras_model.h5')
        if os.path.exists(model_path):
            # Custom objects needed for loading
            custom_objects = {
                'TemporalAttention': TemporalAttention
            }
            self.model = load_model(model_path, custom_objects=custom_objects)
        else:
            logger.warning(f"No Keras model found at {model_path}")


class ModelDevelopment:
    """
    Model development framework for the VMETHOD framework.
    
    This class handles the creation, training, and evaluation of multiple models
    for both single-pair and bagging approaches.
    """
    
    def __init__(self, models_dir: str = 'models/'):
        """
        Initialize the model development framework.
        
        Args:
            models_dir: Directory to save model artifacts
        """
        self.models_dir = models_dir
        
        # Create directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Dictionary to store trained models
        self.models = {}
        
        # Evaluation results
        self.evaluation_results = {}
        
    def create_models(self, approach: str = 'single', currency_pair: str = None) -> Dict[str, BaseModel]:
        """
        Create models for a specific approach and currency pair.
        
        Args:
            approach: Either 'single' or 'bagging'
            currency_pair: Name of currency pair (for single approach)
            
        Returns:
            Dictionary of model instances
        """
        if approach not in ['single', 'bagging']:
            raise ValueError("Approach must be either 'single' or 'bagging'")
            
        # Generate model name prefix
        prefix = f"{approach}_{currency_pair}_" if currency_pair else f"{approach}_"
        
        # Create model instances
        models = {
            'lstm': LSTMModel(name=f"{prefix}LSTM", model_dir=self.models_dir),
            'xgboost': XGBoostModel(name=f"{prefix}XGBoost", model_dir=self.models_dir),
            'gru': GRUModel(name=f"{prefix}GRU", model_dir=self.models_dir),
            'tft': TFTModel(name=f"{prefix}TFT", model_dir=self.models_dir)
        }
        
        # Store models in instance dictionary
        key = f"{approach}_{currency_pair}" if currency_pair else approach
        self.models[key] = models
        
        logger.info(f"Created models for {key}")
        return models
    
    def train_single_approach(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], currency_pairs: List[str]) -> None:
        """
        Train models using the single-pair approach.
        
        Args:
            data_dict: Dictionary of training and testing datasets
            currency_pairs: List of currency pairs to train on
        """
        for pair in currency_pairs:
            logger.info(f"Training models for {pair} (Single approach)")
            
            # Get data for this pair
            if pair not in data_dict:
                logger.error(f"No data found for {pair}")
                continue
                
            # Get train and test data
            train_X = data_dict[pair]['train']['X']
            train_y = data_dict[pair]['train']['y']
            
            # Create models for this pair
            pair_models = self.create_models(approach='single', currency_pair=pair)
            
            # Train each model
            for model_name, model in pair_models.items():
                logger.info(f"Training {model_name} model for {pair}")
                
                try:
                    # Train model with default parameters
                    model.fit(train_X, train_y, verbose=1)
                    
                    # Save trained model
                    model.save()
                    
                    logger.info(f"Successfully trained {model_name} model for {pair}")
                except Exception as e:
                    logger.error(f"Error training {model_name} model for {pair}: {str(e)}")
    
    def train_bagging_approach(self, bagging_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
        """
        Train models using the bagging approach.
        
        Args:
            bagging_data: Dictionary of combined training and testing datasets
        """
        logger.info("Training models using Bagging approach")
        
        # Get train data
        train_X = bagging_data['train']['X']
        train_y = bagging_data['train']['y']
        
        # Create models for bagging approach
        bagging_models = self.create_models(approach='bagging')
        
        # Train each model
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
                
                logger.info(f"Successfully trained {model_name} model (Bagging approach)")
            except Exception as e:
                logger.error(f"Error training {model_name} model (Bagging approach): {str(e)}")
    
    def evaluate_single_approach(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                               currency_pairs: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate models trained with the single-pair approach.
        
        Args:
            data_dict: Dictionary of training and testing datasets
            currency_pairs: List of currency pairs to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation = {}
        
        for pair in currency_pairs:
            logger.info(f"Evaluating models for {pair} (Single approach)")
            
            # Get data for this pair
            if pair not in data_dict:
                logger.error(f"No data found for {pair}")
                continue
                
            # Get test data
            test_X = data_dict[pair]['test']['X']
            test_y = data_dict[pair]['test']['y']
            
            evaluation[pair] = {}
            
            # Get models for this pair
            key = f"single_{pair}"
            if key not in self.models:
                logger.error(f"No models found for {key}")
                continue
                
            # Evaluate each model
            for model_name, model in self.models[key].items():
                if not model.is_trained:
                    logger.warning(f"Model {model_name} for {pair} is not trained")
                    continue
                    
                # Evaluate model
                metrics = model.evaluate(test_X, test_y)
                evaluation[pair][model_name] = metrics
                
                # Log results
                logger.info(f"Evaluation results for {model_name} on {pair}:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value}")
        
        # Store evaluation results
        self.evaluation_results['single'] = evaluation
        
        return evaluation
    
    def evaluate_bagging_approach(self, bagging_data: Dict[str, Dict[str, pd.DataFrame]],
                                currency_pairs: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate models trained with the bagging approach.
        
        Args:
            bagging_data: Dictionary of combined training and testing datasets
            currency_pairs: List of currency pairs to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation = {}
        
        # Get test data
        test_X = bagging_data['test']['X']
        test_y = bagging_data['test']['y']
        
        # For each currency pair in the bagging approach
        for pair in currency_pairs:
            logger.info(f"Evaluating models for {pair} (Bagging approach)")
            
            # Test data for this pair
            pair_test_y = test_y[pair] if pair in test_y.columns else None
            
            if pair_test_y is None:
                logger.error(f"No test data found for {pair} in bagging dataset")
                continue
                
            evaluation[pair] = {}
            
            # Get bagging models
            if 'bagging' not in self.models:
                logger.error("No bagging models found")
                continue
                
            # Evaluate each model
            for model_name, model in self.models['bagging'].items():
                if not model.is_trained:
                    logger.warning(f"Model {model_name} (Bagging approach) is not trained")
                    continue
                    
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
                
                evaluation[pair][model_name] = metrics
                
                # Log results
                logger.info(f"Evaluation results for {model_name} on {pair} (Bagging approach):")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value}")
        
        # Store evaluation results
        self.evaluation_results['bagging'] = evaluation
        
        return evaluation
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of model performance.
        
        Returns:
            DataFrame with performance comparison
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return pd.DataFrame()
            
        # Create a list to store rows
        rows = []
        
        # Add single approach results
        if 'single' in self.evaluation_results:
            for pair, models in self.evaluation_results['single'].items():
                for model_name, metrics in models.items():
                    row = {
                        'Approach': 'Single',
                        'Currency Pair': pair,
                        'Model': model_name
                    }
                    row.update(metrics)
                    rows.append(row)
        
        # Add bagging approach results
        if 'bagging' in self.evaluation_results:
            for pair, models in self.evaluation_results['bagging'].items():
                for model_name, metrics in models.items():
                    row = {
                        'Approach': 'Bagging',
                        'Currency Pair': pair,
                        'Model': model_name
                    }
                    row.update(metrics)
                    rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Round numeric columns for better display
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df
    
    def generate_performance_report(self, save_path: str = 'results/model_performance.csv') -> None:
        """
        Generate and save a performance report.
        
        Args:
            save_path: Path to save the report CSV file
        """
        # Create comparison table
        table = self.create_comparison_table()
        
        if table.empty:
            logger.error("No data available for performance report")
            return
            
        # Save to CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        table.to_csv(save_path, index=False)
        
        logger.info(f"Performance report saved to {save_path}")
    
    def visualize_performance_comparison(self, metric: str = 'rmse', 
                                       save_path: str = 'results/figures/model_comparison.png') -> None:
        """
        Visualize model performance comparison.
        
        Args:
            metric: Metric to visualize
            save_path: Path to save the visualization
        """
        # Create comparison table
        table = self.create_comparison_table()
        
        if table.empty:
            logger.error("No data available for visualization")
            return
            
        # Check if metric exists
        if metric not in table.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get unique currency pairs and models
        currency_pairs = table['Currency Pair'].unique()
        models = table['Model'].unique()
        
        # Set up plot dimensions
        x = np.arange(len(currency_pairs))
        width = 0.2  # Bar width
        
        # Plot bars for each approach and model
        for i, model in enumerate(models):
            # Single approach
            single_data = table[(table['Approach'] == 'Single') & (table['Model'] == model)]
            if not single_data.empty:
                # Align bars side by side
                offset = (i - len(models)/2 + 0.5) * width
                
                # Extract metric values for each currency pair
                values = []
                for pair in currency_pairs:
                    pair_data = single_data[single_data['Currency Pair'] == pair]
                    if not pair_data.empty:
                        values.append(pair_data[metric].values[0])
                    else:
                        values.append(np.nan)
                        
                plt.bar(x + offset, values, width, label=f'Single-{model}')
            
            # Bagging approach
            bagging_data = table[(table['Approach'] == 'Bagging') & (table['Model'] == model)]
            if not bagging_data.empty:
                # Different pattern for bagging approach
                offset = (i - len(models)/2 + 0.5) * width
                
                # Extract metric values for each currency pair
                values = []
                for pair in currency_pairs:
                    pair_data = bagging_data[bagging_data['Currency Pair'] == pair]
                    if not pair_data.empty:
                        values.append(pair_data[metric].values[0])
                    else:
                        values.append(np.nan)
                        
                plt.bar(x + offset, values, width, label=f'Bagging-{model}', alpha=0.7, hatch='///')
                
        # Add labels and title
        plt.xlabel('Currency Pair')
        plt.ylabel(metric.upper())
        plt.title(f'Model Performance Comparison - {metric.upper()}')
        plt.xticks(x, currency_pairs)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Performance comparison visualization saved to {save_path}")