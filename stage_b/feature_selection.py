"""
Feature Selection Framework for the VMETHOD framework.

This module implements methods to identify the most predictive features:
- Random Forest Feature Importance
- Boruta Algorithm Implementation
- SHAP Value Analysis
- Autoencoder Feature Selection
- Feature Fusion System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import shap
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FeatureSelection')

class RandomForestSelector:
    """
    Feature selection using Random Forest feature importance.
    """
    
    def __init__(self):
        """Initialize Random Forest selector."""
        self.feature_importances = {}
        self.models = {}
        
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       threshold: float = 0.01, n_estimators: int = 100, 
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Use Random Forest to select important features.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            threshold: Importance threshold for feature selection
            n_estimators: Number of trees in the forest
            test_size: Proportion of data for testing
            
        Returns:
            DataFrame with selected features, Series with feature importances
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Filter out non-numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # Log the columns that were excluded
        excluded_cols = set(df.columns) - set(numeric_cols) - {target_col}
        if excluded_cols:
            logger.info(f"Excluded non-numeric columns: {excluded_cols}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create and train the model
        logger.info("Training Random Forest model for feature selection...")
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        
        # Select features above threshold
        selected_features = feature_importances[feature_importances >= threshold].index.tolist()
        
        # Evaluate model performance
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Random Forest MSE: {mse:.6f}")
        
        # Store the model and feature importances
        self.models[target_col] = rf
        self.feature_importances[target_col] = feature_importances
        
        logger.info(f"Selected {len(selected_features)} features using Random Forest importance")
        return df[selected_features + [target_col]], feature_importances
    
    def visualize_importances(self, target_col: str, top_n: int = 20, 
                             save_path: Optional[str] = None) -> None:
        """
        Visualize feature importances.
        
        Args:
            target_col: Name of the target column
            top_n: Number of top features to show
            save_path: Path to save visualization
        """
        if target_col not in self.feature_importances:
            logger.error(f"No feature importances found for target {target_col}")
            return
            
        importance = self.feature_importances[target_col]
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = importance.head(top_n)
        
        # Create horizontal bar plot
        ax = sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
        ax.set_title(f'Top {top_n} Features by Random Forest Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance visualization saved to {save_path}")
        else:
            plt.show()


class BorutaSelector:
    """
    Feature selection using the Boruta algorithm.
    """
    
    def __init__(self):
        """Initialize Boruta selector."""
        self.importances = {}
        self.selected_features = {}
        
    def _add_shadow_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add random shadow features to the dataset.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added shadow features
        """
        # Create a copy with shadow features
        X_shadow = X.copy()
        
        # For each column, create shadow feature by shuffling
        for column in X.columns:
            X_shadow[f"shadow_{column}"] = np.random.permutation(X[column].values)
            
        return X_shadow
        
    def select_features(self, df: pd.DataFrame, target_col: str, max_iter: int = 100,
                       test_size: float = 0.2, n_estimators: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Use Boruta algorithm to select relevant features.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            max_iter: Maximum number of iterations
            test_size: Proportion of data for testing
            n_estimators: Number of trees in the random forest
            
        Returns:
            DataFrame with selected features, DataFrame with feature stats
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Filter out non-numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # Log the columns that were excluded
        excluded_cols = set(df.columns) - set(numeric_cols) - {target_col}
        if excluded_cols:
            logger.info(f"Excluded non-numeric columns: {excluded_cols}")

        # Track feature statistics
        feature_stats = pd.DataFrame(index=X.columns)
        feature_stats['accepted'] = False
        feature_stats['rejected'] = False
        feature_stats['importance_history'] = [[] for _ in range(len(X.columns))]
        feature_stats['mean_importance'] = 0.0
        feature_stats['max_importance'] = 0.0
        
        # Iterate until max_iter or all features decided
        for iteration in range(max_iter):
            # If all features are decided, stop
            if (feature_stats['accepted'] | feature_stats['rejected']).all():
                logger.info(f"Boruta converged after {iteration} iterations")
                break
                
            # Add shadow features
            X_shadow = self._add_shadow_features(X)
            
            # Train a random forest
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42+iteration, n_jobs=-1)
            rf.fit(X_shadow, y)
            
            # Get feature importances
            all_importances = pd.Series(rf.feature_importances_, index=X_shadow.columns)
            
            # Split original and shadow features
            orig_importances = all_importances.filter(regex='^(?!shadow_)')
            shadow_importances = all_importances.filter(regex='^shadow_')
            
            # Determine shadow importance threshold (max of shadow importances)
            shadow_threshold = shadow_importances.max()
            
            # Update feature statistics
            for feature in X.columns:
                # Skip already decided features
                if feature_stats.loc[feature, 'accepted'] or feature_stats.loc[feature, 'rejected']:
                    continue
                    
                # Track importance history
                feature_stats.loc[feature, 'importance_history'].append(orig_importances[feature])
                
                # Update mean and max importance
                feature_stats.loc[feature, 'mean_importance'] = np.mean(feature_stats.loc[feature, 'importance_history'])
                feature_stats.loc[feature, 'max_importance'] = np.max(feature_stats.loc[feature, 'importance_history'])
                
                # Accept feature if it's consistently better than shadow features
                if (iteration >= 10) and (all(importance > shadow_threshold for importance in feature_stats.loc[feature, 'importance_history'][-5:])):
                    feature_stats.loc[feature, 'accepted'] = True
                    logger.info(f"Accepted feature {feature} at iteration {iteration}")
                    
                # Reject feature if it's consistently worse than shadow features
                if (iteration >= 10) and (all(importance < shadow_threshold for importance in feature_stats.loc[feature, 'importance_history'][-5:])):
                    feature_stats.loc[feature, 'rejected'] = True
                    logger.info(f"Rejected feature {feature} at iteration {iteration}")
            
            # Log progress
            if (iteration + 1) % 10 == 0:
                accepted = feature_stats['accepted'].sum()
                rejected = feature_stats['rejected'].sum()
                tentative = len(feature_stats) - accepted - rejected
                logger.info(f"Iteration {iteration+1}: {accepted} accepted, {rejected} rejected, {tentative} tentative")
        
        # Get final selected features
        selected_features = feature_stats[feature_stats['accepted']].index.tolist()
        
        # Store results
        self.importances[target_col] = feature_stats
        self.selected_features[target_col] = selected_features
        
        logger.info(f"Boruta selected {len(selected_features)} features")
        return df[selected_features + [target_col]], feature_stats
    
    def visualize_importances(self, target_col: str, save_path: Optional[str] = None) -> None:
        """
        Visualize Boruta feature importances.
        
        Args:
            target_col: Name of the target column
            save_path: Path to save visualization
        """
        if target_col not in self.importances:
            logger.error(f"No Boruta results found for target {target_col}")
            return
            
        feature_stats = self.importances[target_col]
        
        # Sort features by importance
        sorted_stats = feature_stats.sort_values('mean_importance', ascending=False)
        
        # Plot features colored by status
        plt.figure(figsize=(12, max(8, len(sorted_stats) * 0.3)))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(sorted_stats)), sorted_stats['mean_importance'])
        
        # Color bars by status
        for i, (idx, row) in enumerate(sorted_stats.iterrows()):
            if row['accepted']:
                bars[i].set_color('green')
            elif row['rejected']:
                bars[i].set_color('red')
            else:
                bars[i].set_color('gray')
        
        plt.yticks(range(len(sorted_stats)), sorted_stats.index)
        plt.xlabel('Mean Importance')
        plt.title('Boruta Feature Importances')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Accepted'),
            Patch(facecolor='red', label='Rejected'),
            Patch(facecolor='gray', label='Tentative')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Boruta importance visualization saved to {save_path}")
        else:
            plt.show()


class SHAPSelector:
    """
    Feature selection using SHAP values.
    """
    
    def __init__(self):
        """Initialize SHAP selector."""
        self.shap_values = {}
        self.models = {}
        self.selected_features = {}
        
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       threshold: Optional[float] = None, top_n: Optional[int] = None,
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, List[str]]:
        """
        Use SHAP to select important features.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            threshold: Minimum SHAP value to include feature
            top_n: Select top N features
            test_size: Proportion of data for testing
            
        Returns:
            DataFrame with selected features, List of selected features
        """
        if threshold is None and top_n is None:
            threshold = 0.01  # Default threshold
            
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Filter out non-numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # Log the columns that were excluded
        excluded_cols = set(df.columns) - set(numeric_cols) - {target_col}
        if excluded_cols:
            logger.info(f"Excluded non-numeric columns: {excluded_cols}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create and train a model (using RandomForest for SHAP)
        logger.info("Training model for SHAP analysis...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Create explainer and calculate SHAP values
        logger.info("Calculating SHAP values (this may take time)...")
        explainer = shap.TreeExplainer(model)
        
        # Use a sample of data for SHAP values to reduce computation time
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP value for each feature
        if isinstance(shap_values, list):  # For multi-output models
            shap_values = np.abs(shap_values).mean(axis=0)
        else:
            shap_values = np.abs(shap_values).mean(axis=0)
            
        feature_importances = pd.Series(shap_values, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        
        # Select features
        if top_n is not None:
            selected_features = feature_importances.head(top_n).index.tolist()
        else:  # Use threshold
            selected_features = feature_importances[feature_importances >= threshold].index.tolist()
        
        # Store results
        self.models[target_col] = model
        self.shap_values[target_col] = feature_importances
        self.selected_features[target_col] = selected_features
        
        logger.info(f"SHAP selected {len(selected_features)} features")
        return df[selected_features + [target_col]], selected_features
    
    def visualize_importances(self, target_col: str, top_n: int = 20, 
                             save_path: Optional[str] = None) -> None:
        """
        Visualize SHAP feature importances.
        
        Args:
            target_col: Name of the target column
            top_n: Number of top features to show
            save_path: Path to save visualization
        """
        if target_col not in self.shap_values:
            logger.error(f"No SHAP values found for target {target_col}")
            return
            
        importance = self.shap_values[target_col]
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = importance.head(top_n)
        
        # Create horizontal bar plot
        ax = sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
        ax.set_title(f'Top {top_n} Features by SHAP Value')
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP importance visualization saved to {save_path}")
        else:
            plt.show()


class AutoencoderSelector:
    """
    Feature selection using Autoencoder reconstruction weights.
    """
    
    def __init__(self):
        """Initialize Autoencoder selector."""
        self.models = {}
        self.importances = {}
        self.selected_features = {}
        
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       threshold: float = 0.01, encoding_dim: Optional[int] = None,
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Use Autoencoder for feature selection.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            threshold: Minimum importance to include feature
            encoding_dim: Size of the bottleneck layer (default: 1/3 of features)
            test_size: Proportion of data for testing
            
        Returns:
            DataFrame with selected features, Series with feature importances
        """
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Filter out non-numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        X = X[numeric_cols]
        
        # Log the columns that were excluded
        excluded_cols = set(df.columns) - set(numeric_cols) - {target_col}
        if excluded_cols:
            logger.info(f"Excluded non-numeric columns: {excluded_cols}")

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=42)
        
        # Set encoding dimension if not specified
        if encoding_dim is None:
            encoding_dim = max(1, X.shape[1] // 3)
            
        logger.info(f"Building autoencoder with encoding dimension {encoding_dim}")
        
        # Build the autoencoder model
        input_dim = X.shape[1]
        
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder layers
        encoder = Dense(input_dim, activation='relu')(input_layer)
        encoder = Dense(input_dim // 2, activation='relu')(encoder)
        encoder = Dense(encoding_dim, activation='relu')(encoder)
        
        # Decoder layers
        decoder = Dense(input_dim // 2, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='linear')(decoder)
        
        # Create model
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        
        # Compile model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        # Add early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        logger.info("Training autoencoder...")
        autoencoder.fit(
            X_train, X_train,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_data=(X_test, X_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Get feature importances based on reconstruction error
        feature_importances = {}
        
        # For each feature, measure its impact on reconstruction
        for i in range(input_dim):
            # Create a copy of test data with this feature zeroed out
            X_test_modified = X_test.copy()
            X_test_modified[:, i] = 0
            
            # Get reconstruction with the feature zeroed
            reconstructed = autoencoder.predict(X_test_modified, verbose=0)
            
            # Measure reconstruction error
            mse = ((X_test - reconstructed) ** 2).mean()
            
            # Higher error means the feature is more important
            feature_importances[X.columns[i]] = mse
            
        # Convert to Series and normalize
        importance_series = pd.Series(feature_importances)
        importance_series = importance_series / importance_series.max()
        importance_series = importance_series.sort_values(ascending=False)
        
        # Select features above threshold
        selected_features = importance_series[importance_series >= threshold].index.tolist()
        
        # Store results
        self.models[target_col] = autoencoder
        self.importances[target_col] = importance_series
        self.selected_features[target_col] = selected_features
        
        logger.info(f"Autoencoder selected {len(selected_features)} features")
        return df[selected_features + [target_col]], importance_series
    
    def visualize_importances(self, target_col: str, top_n: int = 20, 
                             save_path: Optional[str] = None) -> None:
        """
        Visualize Autoencoder feature importances.
        
        Args:
            target_col: Name of the target column
            top_n: Number of top features to show
            save_path: Path to save visualization
        """
        if target_col not in self.importances:
            logger.error(f"No Autoencoder importances found for target {target_col}")
            return
            
        importance = self.importances[target_col]
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        
        # Get top N features
        top_features = importance.head(top_n)
        
        # Create horizontal bar plot
        ax = sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
        ax.set_title(f'Top {top_n} Features by Autoencoder Importance')
        ax.set_xlabel('Normalized Importance')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Autoencoder importance visualization saved to {save_path}")
        else:
            plt.show()


class FeatureFusionSystem:
    """
    Combine multiple feature selection methods for robust feature selection.
    """
    
    def __init__(self):
        """Initialize feature fusion system."""
        self.random_forest = RandomForestSelector()
        self.boruta = BorutaSelector()
        self.shap = SHAPSelector()
        self.autoencoder = AutoencoderSelector()
        
        self.all_results = {}
        self.fusion_results = {}
        self.selected_features = {}
        
    def apply_all_methods(self, df: pd.DataFrame, target_col: str, 
                         rf_threshold: float = 0.01,
                         shap_top_n: int = 30,
                         autoencoder_threshold: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Apply all feature selection methods.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            rf_threshold: Threshold for Random Forest importance
            shap_top_n: Number of top features for SHAP
            autoencoder_threshold: Threshold for Autoencoder importance
            
        Returns:
            Dictionary with results from each method
        """
        # Apply Random Forest
        logger.info("Applying Random Forest feature selection...")
        rf_selected, rf_importances = self.random_forest.select_features(
            df, target_col, threshold=rf_threshold
        )
        
        # Apply Boruta
        logger.info("Applying Boruta feature selection...")
        boruta_selected, boruta_stats = self.boruta.select_features(
            df, target_col
        )
        
        # Apply SHAP
        logger.info("Applying SHAP feature selection...")
        shap_selected, shap_features = self.shap.select_features(
            df, target_col, top_n=shap_top_n
        )
        
        # Apply Autoencoder
        logger.info("Applying Autoencoder feature selection...")
        try:
            ae_selected, ae_importances = self.autoencoder.select_features(
                df, target_col, threshold=autoencoder_threshold
            )
        except Exception as e:
            logger.warning(f"Autoencoder selection failed: {str(e)}")
            ae_selected = pd.DataFrame()
            ae_importances = pd.Series()
        
        # Store all results
        self.all_results[target_col] = {
            'random_forest': {
                'selected': rf_selected,
                'importances': rf_importances,
                'features': list(rf_selected.columns[:-1])  # Exclude target column
            },
            'boruta': {
                'selected': boruta_selected,
                'stats': boruta_stats,
                'features': list(boruta_selected.columns[:-1])  # Exclude target column
            },
            'shap': {
                'selected': shap_selected,
                'features': shap_features
            },
            'autoencoder': {
                'selected': ae_selected,
                'importances': ae_importances,
                'features': list(ae_selected.columns[:-1]) if not ae_selected.empty else []
            }
        }
        
        logger.info("All feature selection methods completed")
        return self.all_results[target_col]
    
    def fuse_results(self, target_col: str, min_votes: int = 2) -> List[str]:
        """
        Combine results from multiple methods using voting.
        
        Args:
            target_col: Name of the target column
            min_votes: Minimum number of methods that must select a feature
            
        Returns:
            List of selected features after fusion
        """
        if target_col not in self.all_results:
            logger.error(f"No feature selection results found for target {target_col}")
            return []
            
        results = self.all_results[target_col]
        
        # Get all features from each method
        rf_features = set(results['random_forest']['features'])
        boruta_features = set(results['boruta']['features'])
        shap_features = set(results['shap']['features'])
        ae_features = set(results['autoencoder']['features'])
        
        # Count votes for each feature
        all_features = set().union(rf_features, boruta_features, shap_features, ae_features)
        feature_votes = {}
        
        for feature in all_features:
            votes = 0
            if feature in rf_features:
                votes += 1
            if feature in boruta_features:
                votes += 1
            if feature in shap_features:
                votes += 1
            if feature in ae_features:
                votes += 1
                
            feature_votes[feature] = votes
        
        # Select features with at least min_votes
        selected_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        
        # Store fusion results
        self.fusion_results[target_col] = {
            'feature_votes': feature_votes,
            'selected_features': selected_features
        }
        
        self.selected_features[target_col] = selected_features
        
        logger.info(f"Feature fusion selected {len(selected_features)} features with at least {min_votes} votes")
        return selected_features
    
    def visualize_fusion(self, target_col: str, save_path: Optional[str] = None) -> None:
        """
        Visualize feature fusion results.
        
        Args:
            target_col: Name of the target column
            save_path: Path to save visualization
        """
        if target_col not in self.fusion_results:
            logger.error(f"No fusion results found for target {target_col}")
            return
            
        feature_votes = self.fusion_results[target_col]['feature_votes']
        
        # Convert to DataFrame for easier plotting
        vote_df = pd.DataFrame({'Feature': list(feature_votes.keys()), 
                               'Votes': list(feature_votes.values())})
        
        # Sort by votes (descending) then by feature name
        vote_df = vote_df.sort_values(['Votes', 'Feature'], ascending=[False, True])
        
        # Create color map for votes
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        plt.figure(figsize=(14, max(8, len(vote_df) * 0.3)))
        
        # Create horizontal bar plot
        bars = plt.barh(vote_df['Feature'], vote_df['Votes'])
        
        # Color bars by vote count
        for i, bar in enumerate(bars):
            bar.set_color(colors[int(vote_df['Votes'].iloc[i]) - 1])
        
        plt.xlabel('Number of Votes')
        plt.ylabel('Feature')
        plt.title('Feature Fusion Results')
        plt.xlim(0, 4.5)  # Max 4 votes possible
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label='1 Vote'),
            Patch(facecolor=colors[1], label='2 Votes'),
            Patch(facecolor=colors[2], label='3 Votes'),
            Patch(facecolor=colors[3], label='4 Votes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature fusion visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_selected_features(self, df: pd.DataFrame, target_col: str, 
                              output_dir: str = 'data/selected_features/') -> pd.DataFrame:
        """
        Save selected features to CSV.
        
        Args:
            df: Original DataFrame
            target_col: Name of the target column
            output_dir: Directory to save selected features
            
        Returns:
            DataFrame with only selected features
        """
        if target_col not in self.selected_features:
            logger.error(f"No selected features found for target {target_col}")
            return df
            
        selected_features = self.selected_features[target_col]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame with only selected features
        selected_df = df[selected_features + [target_col]]
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{target_col}_selected_features.csv")
        selected_df.to_csv(output_path, index=True)
        logger.info(f"Saved selected features for {target_col} to {output_path}")
        
        return selected_df


class FeatureSelectionFramework:
    """
    Comprehensive framework for feature selection in the VMETHOD framework.
    """
    
    def __init__(self):
        """Initialize feature selection framework."""
        self.fusion_system = FeatureFusionSystem()
        self.selected_data = {}
        
    def select_features(self, data_dict: Dict[str, pd.DataFrame], target_col: str, 
                       min_votes: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Apply feature selection to multiple currency pairs.
        
        Args:
            data_dict: Dictionary of DataFrames with features and targets
            target_col: Name of the target column
            min_votes: Minimum votes for feature fusion
            
        Returns:
            Dictionary of DataFrames with selected features
        """
        for pair_name, df in data_dict.items():
            logger.info(f"Selecting features for {pair_name}")
            
            # Convert column names to strings if needed
            df.columns = df.columns.astype(str)
            
            # Make sure target exists in the DataFrame
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found for {pair_name}")
                continue
                
            # Apply all feature selection methods
            self.fusion_system.apply_all_methods(df, target_col)
            
            # Fuse results
            selected_features = self.fusion_system.fuse_results(target_col, min_votes)
            
            # Get DataFrame with selected features
            if selected_features:
                self.selected_data[pair_name] = df[selected_features + [target_col]]
            else:
                logger.warning(f"No features selected for {pair_name}")
                self.selected_data[pair_name] = df  # Keep all features if none selected
                
        return self.selected_data
    
    def save_selected_features(self, output_dir: str = 'data/selected_features/') -> None:
        """
        Save selected features to CSV files.
        
        Args:
            output_dir: Directory to save selected features
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for pair_name, df in self.selected_data.items():
            output_path = os.path.join(output_dir, f"{pair_name}_selected_features.csv")
            df.to_csv(output_path, index=True)
            logger.info(f"Saved selected features for {pair_name} to {output_path}")
    
    def create_importance_visualizations(self, target_col: str, 
                                        output_dir: str = 'results/figures/feature_selection/') -> None:
        """
        Create visualizations for feature importance and selection.
        
        Args:
            target_col: Name of the target column
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual method visualizations
        self.fusion_system.random_forest.visualize_importances(
            target_col, 
            save_path=os.path.join(output_dir, f"random_forest_importance_{target_col}.png")
        )
        
        self.fusion_system.boruta.visualize_importances(
            target_col, 
            save_path=os.path.join(output_dir, f"boruta_importance_{target_col}.png")
        )
        
        self.fusion_system.shap.visualize_importances(
            target_col, 
            save_path=os.path.join(output_dir, f"shap_importance_{target_col}.png")
        )
        
        self.fusion_system.autoencoder.visualize_importances(
            target_col, 
            save_path=os.path.join(output_dir, f"autoencoder_importance_{target_col}.png")
        )
        
        # Create fusion visualization
        self.fusion_system.visualize_fusion(
            target_col,
            save_path=os.path.join(output_dir, f"feature_fusion_{target_col}.png")
        )
        
        logger.info(f"Created feature selection visualizations for {target_col}")