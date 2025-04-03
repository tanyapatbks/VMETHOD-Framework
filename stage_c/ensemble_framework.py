"""
Ensemble Framework for the VMETHOD framework.

This module implements ensemble methods to combine multiple models for more robust predictions.
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
from copy import deepcopy

# ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Import base models
from .model_development import BaseModel, LSTMModel, XGBoostModel, GRUModel, TFTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EnsembleFramework')


class EnsembleFramework:
    """
    Ensemble Framework for combining multiple prediction models.
    
    This class implements methods to combine predictions from multiple models
    for more robust forecasting.
    """
    
    def __init__(self, model_dir: str = 'models/ensemble/'):
        """
        Initialize the Ensemble Framework.
        
        Args:
            model_dir: Directory to save ensemble models
        """
        self.model_dir = model_dir
        self.base_models = {}
        self.meta_model = None
        self.scaler = None
        self.is_trained = False
        self.ensemble_weights = None
        self.dynamic_weights = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def add_model(self, model: BaseModel, name: Optional[str] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: BaseModel instance to add
            name: Name for the model (if None, use model.name)
        """
        if not model.is_trained:
            logger.warning(f"Model {model.name} is not trained. Train it before adding to ensemble.")
            return
            
        model_name = name if name else model.name
        self.base_models[model_name] = model
        
        logger.info(f"Added model {model_name} to ensemble")
    
    def weighted_average_ensemble(self, X: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Generate predictions using weighted average of base models.
        
        Args:
            X: Feature DataFrame
            weights: Dictionary of model weights (if None, use equal weights)
            
        Returns:
            Numpy array of ensemble predictions
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble. Add models first.")
            
        # If no weights provided, use equal weights
        if weights is None:
            weights = {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}
        
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        weights = {name: w / weight_sum for name, w in weights.items()}
        
        # Generate predictions from each model
        predictions = {}
        for name, model in self.base_models.items():
            if name in weights and weights[name] > 0:
                predictions[name] = model.predict(X)
        
        # Combine predictions
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return ensemble_pred
    
    def train_stacking_ensemble(self, X: pd.DataFrame, y: pd.DataFrame,
                              meta_model: str = 'linear', cv_folds: int = 5) -> None:
        """
        Train a stacking ensemble using cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            meta_model: Type of meta-model ('linear', 'ridge', 'lasso', 'rf', 'gbm')
            cv_folds: Number of cross-validation folds
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble. Add models first.")
            
        logger.info(f"Training stacking ensemble with {meta_model} meta-model and {cv_folds}-fold CV")
        
        # Convert y to numpy array
        y_array = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Split data into folds
        n_samples = len(X)
        fold_size = n_samples // cv_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Create matrix to store out-of-fold predictions
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        # Generate out-of-fold predictions for each model
        for fold in range(cv_folds):
            logger.info(f"Processing fold {fold+1}/{cv_folds}")
            
            # Define train and validation indices for this fold
            val_indices = indices[fold * fold_size:(fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)
            
            # Extract train and validation sets
            X_train, X_val = X.iloc[train_indices].copy(), X.iloc[val_indices].copy()
            y_train, y_val = y_array[train_indices], y_array[val_indices]
            
            # Generate predictions for validation set from each model
            for i, (name, model) in enumerate(self.base_models.items()):
                # Clone model to avoid retraining the original
                model_clone = deepcopy(model)
                
                # Train on this fold's training data
                if isinstance(model, (LSTMModel, GRUModel, TFTModel)):
                    # For neural network models, train for fewer epochs
                    model_clone.fit(X_train, pd.DataFrame(y_train), epochs=50, verbose=0)
                else:
                    model_clone.fit(X_train, pd.DataFrame(y_train))
                
                # Predict on validation data
                meta_features[val_indices, i] = model_clone.predict(X_val)
        
        # Scale meta-features for better meta-model training
        self.scaler = StandardScaler()
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Create and train meta-model
        if meta_model == 'linear':
            self.meta_model = LinearRegression()
        elif meta_model == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif meta_model == 'lasso':
            self.meta_model = Lasso(alpha=0.1)
        elif meta_model == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif meta_model == 'gbm':
            self.meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown meta-model type: {meta_model}")
            
        self.meta_model.fit(meta_features_scaled, y_array)
        
        # Set model weights based on meta-model coefficients (for linear models)
        if meta_model in ['linear', 'ridge', 'lasso']:
            self.ensemble_weights = {}
            for i, (name, _) in enumerate(self.base_models.items()):
                self.ensemble_weights[name] = abs(self.meta_model.coef_[i])
                
            # Normalize weights
            weight_sum = sum(self.ensemble_weights.values())
            if weight_sum > 0:
                self.ensemble_weights = {name: w / weight_sum for name, w in self.ensemble_weights.items()}
                
        self.is_trained = True
        
        logger.info(f"Stacking ensemble trained successfully")
        
        # Evaluate ensemble on training data
        ensemble_pred = self.predict_stacking(X)
        mse = mean_squared_error(y_array, ensemble_pred)
        logger.info(f"Ensemble training MSE: {mse:.6f}")
    
    def predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the stacking ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of ensemble predictions
        """
        if not self.is_trained or self.meta_model is None:
            raise ValueError("Stacking ensemble not trained. Call train_stacking_ensemble first.")
            
        # Generate predictions from each base model
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            meta_features[:, i] = model.predict(X)
            
        # Scale meta-features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Generate ensemble predictions
        ensemble_pred = self.meta_model.predict(meta_features_scaled)
        
        return ensemble_pred
    
    def train_dynamic_weighting(self, X: pd.DataFrame, y: pd.DataFrame,
                              window_size: int = 20, error_metric: str = 'rmse') -> None:
        """
        Train dynamic weighting based on recent performance.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            window_size: Size of window to consider for recent performance
            error_metric: Metric to use for weighting ('rmse', 'mae', 'mape')
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble. Add models first.")
            
        logger.info(f"Training dynamic weighting with window size {window_size}")
        
        # Convert y to numpy array
        y_array = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Generate predictions from each model
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
            
        # Calculate errors for each sample
        errors = {}
        for name, pred in predictions.items():
            if error_metric == 'rmse':
                errors[name] = np.sqrt((y_array - pred) ** 2)
            elif error_metric == 'mae':
                errors[name] = np.abs(y_array - pred)
            elif error_metric == 'mape':
                errors[name] = np.abs((y_array - pred) / y_array) * 100
            else:
                raise ValueError(f"Unknown error metric: {error_metric}")
                
        # Initialize dynamic weights storage
        self.dynamic_weights = {
            'window_size': window_size,
            'error_metric': error_metric,
            'model_errors': errors
        }
        
        self.is_trained = True
        
        logger.info(f"Dynamic weighting trained successfully")
    
    def predict_dynamic(self, X: pd.DataFrame, current_index: int) -> np.ndarray:
        """
        Generate predictions using dynamic weighting.
        
        Args:
            X: Feature DataFrame
            current_index: Current time index (for calculating weights based on recent performance)
            
        Returns:
            Numpy array of ensemble predictions
        """
        if not self.is_trained or not self.dynamic_weights:
            raise ValueError("Dynamic weighting not trained. Call train_dynamic_weighting first.")
            
        # Get window size and errors
        window_size = self.dynamic_weights['window_size']
        errors = self.dynamic_weights['model_errors']
        
        # Calculate window start index (ensure it's not negative)
        window_start = max(0, current_index - window_size)
        
        # Calculate recent performance for each model
        recent_performance = {}
        for name, error in errors.items():
            # Skip if we don't have enough history
            if current_index < window_size or window_start >= len(error):
                # Use equal weights if not enough history
                recent_performance[name] = 1.0
                continue
                
            # Get recent errors
            recent_errors = error[window_start:current_index]
            
            # Calculate mean error
            mean_error = np.mean(recent_errors)
            
            # Invert (lower error = higher weight)
            recent_performance[name] = 1.0 / (mean_error + 1e-10)
            
        # Normalize weights to sum to 1
        weight_sum = sum(recent_performance.values())
        weights = {name: perf / weight_sum for name, perf in recent_performance.items()}
        
        # Generate predictions from each model
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
            
        # Combine predictions with dynamic weights
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return ensemble_pred
    
    def train_regime_specific(self, X: pd.DataFrame, y: pd.DataFrame, 
                            regime_col: str, regimes: List[Any]) -> None:
        """
        Train regime-specific models for different market conditions.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            regime_col: Column name containing regime labels
            regimes: List of possible regime values
        """
        if not self.base_models:
            raise ValueError("No base models in ensemble. Add models first.")
            
        if regime_col not in X.columns:
            raise ValueError(f"Regime column '{regime_col}' not found in features")
            
        logger.info(f"Training regime-specific ensemble for {len(regimes)} different regimes")
        
        # Convert y to numpy array
        y_array = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Initialize regime-specific weights
        regime_weights = {}
        
        # For each regime, find the best performing model
        for regime in regimes:
            logger.info(f"Analyzing regime {regime}")
            
            # Get indices for this regime
            regime_indices = X[regime_col] == regime
            
            # Skip if no samples for this regime
            if not any(regime_indices):
                logger.warning(f"No samples found for regime {regime}")
                regime_weights[regime] = {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}
                continue
                
            # Extract data for this regime
            X_regime = X[regime_indices].copy().drop(columns=[regime_col])
            y_regime = y_array[regime_indices]
            
            # Calculate performance for each model
            performance = {}
            for name, model in self.base_models.items():
                # Generate predictions
                pred = model.predict(X_regime)
                
                # Calculate error
                mse = mean_squared_error(y_regime, pred)
                
                # Store inverted error (lower error = higher weight)
                performance[name] = 1.0 / (mse + 1e-10)
                
            # Normalize weights
            weight_sum = sum(performance.values())
            regime_weights[regime] = {name: perf / weight_sum for name, perf in performance.items()}
            
        # Store regime weights
        self.ensemble_weights = regime_weights
        self.is_trained = True
        
        logger.info(f"Regime-specific ensemble trained successfully")
    
    def predict_regime_specific(self, X: pd.DataFrame, regime_col: str) -> np.ndarray:
        """
        Generate predictions using regime-specific weighting.
        
        Args:
            X: Feature DataFrame
            regime_col: Column name containing regime labels
            
        Returns:
            Numpy array of ensemble predictions
        """
        if not self.is_trained or not self.ensemble_weights:
            raise ValueError("Regime-specific ensemble not trained. Call train_regime_specific first.")
            
        if regime_col not in X.columns:
            raise ValueError(f"Regime column '{regime_col}' not found in features")
            
        # Generate predictions from each model
        base_predictions = {}
        for name, model in self.base_models.items():
            # Use a version of X without the regime column for prediction
            X_pred = X.drop(columns=[regime_col])
            base_predictions[name] = model.predict(X_pred)
            
        # Initialize ensemble predictions
        ensemble_pred = np.zeros(len(X))
        
        # Apply regime-specific weights
        for i, regime in enumerate(X[regime_col]):
            # Get weights for this regime
            if regime in self.ensemble_weights:
                weights = self.ensemble_weights[regime]
            else:
                # Use equal weights if regime not seen during training
                weights = {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}
                
            # Combine predictions for this sample
            for name, pred in base_predictions.items():
                ensemble_pred[i] += weights[name] * pred[i]
                
        return ensemble_pred
    
    def save(self, save_path: Optional[str] = None) -> str:
        """
        Save the ensemble framework.
        
        Args:
            save_path: Path to save the ensemble (if None, generate based on timestamp)
            
        Returns:
            Path where ensemble was saved
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.model_dir, f"ensemble_{timestamp}")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'is_trained': self.is_trained,
            'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'base_models': list(self.base_models.keys())
        }
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save scaler if available
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(save_path, 'scaler.joblib'))
            
        # Save meta-model if available
        if self.meta_model is not None:
            joblib.dump(self.meta_model, os.path.join(save_path, 'meta_model.joblib'))
            
        # Save ensemble weights if available
        if self.ensemble_weights is not None:
            with open(os.path.join(save_path, 'ensemble_weights.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                weights_json = {}
                
                # Handle different weight structures
                if isinstance(self.ensemble_weights, dict):
                    # Check if values are dictionaries (regime-specific weights)
                    if all(isinstance(v, dict) for v in self.ensemble_weights.values()):
                        for regime, weights in self.ensemble_weights.items():
                            weights_json[str(regime)] = weights
                    else:
                        # Simple weights
                        weights_json = self.ensemble_weights
                        
                json.dump(weights_json, f, indent=2)
                
        # Save dynamic weights if available
        if self.dynamic_weights:
            # Filter out non-serializable parts
            serializable_weights = {
                'window_size': self.dynamic_weights['window_size'],
                'error_metric': self.dynamic_weights['error_metric']
            }
            
            with open(os.path.join(save_path, 'dynamic_weights.json'), 'w') as f:
                json.dump(serializable_weights, f, indent=2)
                
        logger.info(f"Ensemble framework saved to {save_path}")
        return save_path
    
    def load(self, load_path: str) -> None:
        """
        Load the ensemble framework.
        
        Args:
            load_path: Path to load the ensemble from
        """
        # Load metadata
        with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.is_trained = metadata['is_trained']
        
        # Load scaler if available
        scaler_path = os.path.join(load_path, 'scaler.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        # Load meta-model if available
        meta_model_path = os.path.join(load_path, 'meta_model.joblib')
        if os.path.exists(meta_model_path):
            self.meta_model = joblib.load(meta_model_path)
            
        # Load ensemble weights if available
        weights_path = os.path.join(load_path, 'ensemble_weights.json')
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)
                
        # Load dynamic weights if available
        dynamic_weights_path = os.path.join(load_path, 'dynamic_weights.json')
        if os.path.exists(dynamic_weights_path):
            with open(dynamic_weights_path, 'r') as f:
                self.dynamic_weights = json.load(f)
                
        logger.info(f"Ensemble framework loaded from {load_path}")
    
    def plot_model_weights(self, regime: Optional[Any] = None, 
                         save_path: Optional[str] = None) -> None:
        """
        Plot model weights in the ensemble.
        
        Args:
            regime: Specific regime to plot weights for (for regime-specific ensembles)
            save_path: Path to save the plot
        """
        if not self.is_trained or not self.ensemble_weights:
            logger.warning("Ensemble not trained or no weights available")
            return
            
        # Extract weights for visualization
        if isinstance(self.ensemble_weights, dict):
            # Check if values are dictionaries (regime-specific weights)
            if all(isinstance(v, dict) for v in self.ensemble_weights.values()):
                if regime is None:
                    logger.warning("Please specify a regime for regime-specific ensemble")
                    return
                    
                # Extract weights for the specified regime
                if str(regime) in self.ensemble_weights:
                    weights = self.ensemble_weights[str(regime)]
                elif regime in self.ensemble_weights:
                    weights = self.ensemble_weights[regime]
                else:
                    logger.warning(f"Regime {regime} not found in ensemble weights")
                    return
            else:
                # Simple weights
                weights = self.ensemble_weights
        else:
            logger.warning("Unexpected weight format")
            return
            
        # Convert weights to Series for plotting
        weights_series = pd.Series(weights)
        
        # Sort by weight
        weights_series = weights_series.sort_values(ascending=False)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot horizontal bar chart
        weights_series.plot(kind='barh', color='skyblue')
        
        title = "Ensemble Model Weights"
        if regime is not None:
            title += f" (Regime: {regime})"
            
        plt.title(title)
        plt.xlabel('Weight')
        plt.ylabel('Model')
        plt.grid(True, alpha=0.3)
        
        # Add weight values as text
        for i, weight in enumerate(weights_series):
            plt.text(weight + 0.01, i, f"{weight:.4f}", va='center')
            
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model weights plot saved to {save_path}")
        else:
            plt.show()
            
    def analyze_ensemble_diversity(self, X: pd.DataFrame, y: pd.DataFrame,
                                 save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze the diversity of ensemble models.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame or Series
            save_path: Path to save the diversity plot
            
        Returns:
            Dictionary of diversity metrics
        """
        if not self.base_models:
            logger.warning("No base models in ensemble")
            return {}
            
        # Convert y to numpy array
        y_array = y.values.flatten() if isinstance(y, pd.DataFrame) else y.values
        
        # Generate predictions from each model
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
            
        # Calculate errors for each model
        errors = {}
        for name, pred in predictions.items():
            errors[name] = y_array - pred
            
        # Convert errors to DataFrame for correlation analysis
        errors_df = pd.DataFrame(errors)
        
        # Calculate correlation between model errors
        error_corr = errors_df.corr()
        
        # Calculate diversity metrics
        # Lower correlation = higher diversity
        diversity = {}
        
        # Average pairwise error correlation (lower is better)
        # Exclude diagonal elements (self-correlation)
        n_models = len(self.base_models)
        avg_corr = (error_corr.sum().sum() - n_models) / (n_models * (n_models - 1))
        diversity['avg_error_correlation'] = avg_corr
        
        # Create heatmap of error correlations
        plt.figure(figsize=(10, 8))
        
        # Plot correlation matrix
        im = plt.imshow(error_corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, label='Error Correlation')
        
        # Add labels
        plt.xticks(range(len(error_corr)), error_corr.columns, rotation=45, ha='right')
        plt.yticks(range(len(error_corr)), error_corr.index)
        
        # Add correlation values as text
        for i in range(len(error_corr)):
            for j in range(len(error_corr)):
                plt.text(j, i, f"{error_corr.iloc[i, j]:.2f}", 
                       ha='center', va='center', 
                       color='white' if abs(error_corr.iloc[i, j]) > 0.5 else 'black')
        
        plt.title(f"Model Error Correlation\nAverage: {avg_corr:.4f}")
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ensemble diversity plot saved to {save_path}")
        else:
            plt.show()
            
        return diversity