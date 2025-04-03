"""
Training Strategy Implementation for the VMETHOD framework.

This module handles data splitting, single and bagging forecast frameworks,
and evaluation methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainingStrategy')

class TrainingStrategy:
    """
    Training Strategy Implementation for the VMETHOD framework.
    
    This class handles data splitting, single and bagging forecast frameworks,
    and evaluation methodology.
    """
    
    def __init__(self, output_dir: str = 'data/training/'):
        """
        Initialize the Training Strategy.
        
        Args:
            output_dir: Directory to save training datasets
        """
        self.output_dir = output_dir
        self.splits = {}  # Store train-test splits
        self.single_datasets = {}  # Store single-pair datasets
        self.bagging_datasets = {}  # Store bagging datasets
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def create_time_splits(self, data_dict: Dict[str, pd.DataFrame], 
                          train_years: List[str], test_period: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create chronological train-test splits based on years.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            train_years: List of years to include in training data (e.g., ['2020', '2021'])
            test_period: Period for testing data (e.g., '2022-01/2022-03')
            
        Returns:
            Dictionary of train-test splits for each currency pair
        """
        for pair_name, df in data_dict.items():
            # Create train-test split for this pair
            train_mask = False
            for year in train_years:
                # Convert the index's strftime result to a Series, then apply eq()
                train_mask |= pd.Series(df.index.strftime('%Y'), index=df.index).eq(year)
                
            # Parse test period
            if '/' in test_period:
                test_start, test_end = test_period.split('/')
            else:
                test_start = test_period
                # If only start is provided, assume 3 months test period
                year, month = test_start.split('-')
                month = int(month)
                end_month = (month + 2) % 12 + 1  # 3 months later
                end_year = int(year) + (month + 2) // 12  # Adjust year if needed
                test_end = f"{end_year}-{end_month:02d}"
            
            test_mask = ((df.index >= test_start) & (df.index < test_end))
            
            # Store train and test sets
            self.splits[pair_name] = {
                'train': df[train_mask].copy(),
                'test': df[test_mask].copy()
            }
            
            logger.info(f"Created time split for {pair_name}: "
                      f"Train: {len(self.splits[pair_name]['train'])} rows, "
                      f"Test: {len(self.splits[pair_name]['test'])} rows")
            
        return self.splits
    
    def create_single_forecast_datasets(self, column: str = 'Close', 
                                      horizon: int = 1, window: int = 20) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Create datasets for single-pair forecasting.
        
        Args:
            column: Target column to predict
            horizon: Forecast horizon (number of days ahead to predict)
            window: Lookback window (number of days of history to use)
            
        Returns:
            Dictionary of feature and target datasets for each pair
        """
        if not self.splits:
            logger.error("No train-test splits available. Call create_time_splits first.")
            return {}
        
        for pair_name, split_data in self.splits.items():
            train_df = split_data['train']
            test_df = split_data['test']
            
            # Prepare datasets for this pair
            self.single_datasets[pair_name] = {}
            
            # Process train and test sets
            for set_name, df in [('train', train_df), ('test', test_df)]:
                # Create target (shifted future values)
                target = df[column].shift(-horizon)
                
                # Create features
                features = pd.DataFrame(index=df.index)
                
                # Historical values as features
                for i in range(window):
                    features[f'{column}_lag_{i+1}'] = df[column].shift(i)
                
                # Add any processed features from the DataFrame
                prefix = f'{column}_'
                processed_cols = [c for c in df.columns if c.startswith(prefix) and c != column]
                
                for col in processed_cols:
                    features[col] = df[col]
                
                # Align features and target, and drop NaN rows
                combined = pd.concat([features, target.rename(f'{column}_target')], axis=1)
                combined = combined.dropna()
                
                # Split back into features and target
                X = combined.drop(f'{column}_target', axis=1)
                y = combined[f'{column}_target']
                
                # Store datasets
                self.single_datasets[pair_name][set_name] = {
                    'X': X,
                    'y': y
                }
                
            # Save datasets to disk
            train_X_path = os.path.join(self.output_dir, f'{pair_name}_single_train_X.csv')
            train_y_path = os.path.join(self.output_dir, f'{pair_name}_single_train_y.csv')
            test_X_path = os.path.join(self.output_dir, f'{pair_name}_single_test_X.csv')
            test_y_path = os.path.join(self.output_dir, f'{pair_name}_single_test_y.csv')
            
            self.single_datasets[pair_name]['train']['X'].to_csv(train_X_path)
            self.single_datasets[pair_name]['train']['y'].to_csv(train_y_path)
            self.single_datasets[pair_name]['test']['X'].to_csv(test_X_path)
            self.single_datasets[pair_name]['test']['y'].to_csv(test_y_path)
            
            logger.info(f"Created and saved single forecast datasets for {pair_name}")
            
        return self.single_datasets
    
    def create_bagging_forecast_datasets(self, column: str = 'Close', 
                                       horizon: int = 1, window: int = 20) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create datasets for bagging forecast approach (combining all pairs).
        
        Args:
            column: Target column to predict
            horizon: Forecast horizon (number of days ahead to predict)
            window: Lookback window (number of days of history to use)
            
        Returns:
            Dictionary of combined feature and target datasets
        """
        if not self.splits:
            logger.error("No train-test splits available. Call create_time_splits first.")
            return {}
        
        # Lists to store DataFrames from all pairs
        train_X_all = []
        train_y_all = []
        test_X_all = []
        test_y_all = []
        
        # Process each pair
        for pair_name, split_data in self.splits.items():
            train_df = split_data['train']
            test_df = split_data['test']
            
            # Process train and test sets
            for set_name, df in [('train', train_df), ('test', test_df)]:
                # Create target (shifted future values)
                target = df[column].shift(-horizon)
                
                # Create features
                features = pd.DataFrame(index=df.index)
                
                # Historical values as features
                for i in range(window):
                    features[f'{pair_name}_{column}_lag_{i+1}'] = df[column].shift(i)
                
                # Add any processed features from the DataFrame with pair name prefix
                prefix = f'{column}_'
                processed_cols = [c for c in df.columns if c.startswith(prefix) and c != column]
                
                for col in processed_cols:
                    features[f'{pair_name}_{col}'] = df[col]
                
                # Align features and target, and drop NaN rows
                combined = pd.concat([features, target.rename(f'{pair_name}_{column}_target')], axis=1)
                combined = combined.dropna()
                
                # Split back into features and target
                X = combined.drop(f'{pair_name}_{column}_target', axis=1)
                y = combined[f'{pair_name}_{column}_target']
                
                # Add to the appropriate lists
                if set_name == 'train':
                    train_X_all.append(X)
                    train_y_all.append(y.rename(pair_name))
                else:
                    test_X_all.append(X)
                    test_y_all.append(y.rename(pair_name))
        
        # Combine data from all pairs
        if train_X_all and train_y_all and test_X_all and test_y_all:
            # Combine features
            train_X_combined = pd.concat(train_X_all, axis=1)
            test_X_combined = pd.concat(test_X_all, axis=1)
            
            # Combine targets (as separate columns)
            train_y_combined = pd.concat(train_y_all, axis=1)
            test_y_combined = pd.concat(test_y_all, axis=1)
            
            # Store the bagging datasets
            self.bagging_datasets = {
                'train': {
                    'X': train_X_combined,
                    'y': train_y_combined
                },
                'test': {
                    'X': test_X_combined,
                    'y': test_y_combined
                }
            }
            
            # Save datasets to disk
            train_X_path = os.path.join(self.output_dir, 'bagging_train_X.csv')
            train_y_path = os.path.join(self.output_dir, 'bagging_train_y.csv')
            test_X_path = os.path.join(self.output_dir, 'bagging_test_X.csv')
            test_y_path = os.path.join(self.output_dir, 'bagging_test_y.csv')
            
            train_X_combined.to_csv(train_X_path)
            train_y_combined.to_csv(train_y_path)
            test_X_combined.to_csv(test_X_path)
            test_y_combined.to_csv(test_y_path)
            
            logger.info(f"Created and saved bagging forecast datasets")
            
        else:
            logger.error("Failed to create bagging datasets - no data available")
            
        return self.bagging_datasets
    
    def visualize_all_pairs(self, data_dict: Dict[str, pd.DataFrame], column: str = 'Close',
                           start_date: Optional[str] = None, end_date: Optional[str] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize all currency pairs together on the same graph.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            column: Column to plot
            start_date: Start date for visualization (YYYY-MM-DD)
            end_date: End date for visualization (YYYY-MM-DD)
            save_path: Path to save the visualization, if None, display only
        """
        if not data_dict:
            logger.error("No data provided for visualization")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Dictionary to store normalized data for each pair
        normalized_data = {}
        
        # Normalize each pair to start from 100 for better comparison
        for pair_name, df in data_dict.items():
            # Filter by date range if provided
            plot_df = df.copy()
            if start_date:
                plot_df = plot_df[plot_df.index >= start_date]
            if end_date:
                plot_df = plot_df[plot_df.index <= end_date]
                
            # Normalize to start from 100
            first_value = plot_df[column].iloc[0]
            normalized_data[pair_name] = plot_df[column] / first_value * 100
            
            # Plot normalized data
            plt.plot(plot_df.index, normalized_data[pair_name], label=pair_name)
        
        plt.title('Normalized Price Comparison (All Currency Pairs)')
        plt.ylabel('Normalized Price (Base=100)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"All pairs visualization saved to {save_path}")
        else:
            plt.show()
            
    def evaluate_approaches(self, save_path: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Compare dataset characteristics between single and bagging approaches.
        
        Args:
            save_path: Path to save the comparison visualization
            
        Returns:
            Dictionary with dataset statistics
        """
        if not self.single_datasets or not self.bagging_datasets:
            logger.error("Both single and bagging datasets must be created first")
            return {}
        
        comparison = {
            'single': {},
            'bagging': {}
        }
        
        # Analyze single approach datasets
        for pair_name, datasets in self.single_datasets.items():
            comparison['single'][pair_name] = {
                'train_samples': len(datasets['train']['X']),
                'train_features': datasets['train']['X'].shape[1],
                'test_samples': len(datasets['test']['X']),
                'test_features': datasets['test']['X'].shape[1]
            }
        
        # Analyze bagging approach datasets
        comparison['bagging'] = {
            'train_samples': len(self.bagging_datasets['train']['X']),
            'train_features': self.bagging_datasets['train']['X'].shape[1],
            'test_samples': len(self.bagging_datasets['test']['X']),
            'test_features': self.bagging_datasets['test']['X'].shape[1]
        }
        
        # Visualize comparison
        if save_path:
            plt.figure(figsize=(10, 6))
            
            # Plot number of features
            plt.subplot(1, 2, 1)
            
            # For single approach, average across pairs
            single_train_features = [stats['train_features'] for pair, stats in comparison['single'].items()]
            single_avg_features = np.mean(single_train_features)
            
            feature_data = [single_avg_features, comparison['bagging']['train_features']]
            plt.bar(['Single Approach\n(Avg per Pair)', 'Bagging Approach'], feature_data)
            plt.title('Number of Features')
            plt.ylabel('Feature Count')
            
            # Plot sample sizes
            plt.subplot(1, 2, 2)
            
            # For single approach, show each pair separately
            pairs = list(comparison['single'].keys())
            single_train_samples = [stats['train_samples'] for pair, stats in comparison['single'].items()]
            bagging_train_samples = comparison['bagging']['train_samples']
            
            # Create positions for bars
            x_pos = np.arange(len(pairs) + 1)
            
            plt.bar(x_pos[:-1], single_train_samples, label='Single Approach')
            plt.bar(x_pos[-1], bagging_train_samples, label='Bagging Approach')
            
            plt.xticks(x_pos, pairs + ['Bagging'])
            plt.title('Training Sample Size')
            plt.ylabel('Number of Samples')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Approach comparison visualization saved to {save_path}")
        
        return comparison