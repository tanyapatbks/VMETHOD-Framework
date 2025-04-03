"""
Data Acquisition System for the VMETHOD framework.

This module handles loading, validating, and visualizing forex data from CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime
import mplfinance as mpf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataAcquisition')

class DataAcquisitionSystem:
    """
    Data Acquisition System for the VMETHOD framework.
    
    This class handles loading, validating, and visualizing forex data.
    """
    
    def __init__(self, data_dir: str = 'data/raw/'):
        """
        Initialize the Data Acquisition System.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        self.data = {}  # Dictionary to store dataframes for each currency pair
        self.validation_report = {}  # Store validation results
        
    def load_csv_data(self, files: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load historical price data from CSV files.
        
        Args:
            files: List of CSV files to load. If None, load all CSV files in data_dir
            
        Returns:
            Dictionary of DataFrames with currency pair names as keys
        """
        if files is None:
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            
            try:
                # Extract currency pair name from filename
                pair_name = os.path.splitext(file)[0].replace('_1D', '')
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Standardize column names
                df.columns = [col.strip().capitalize() for col in df.columns]
                
                # Ensure required columns exist
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"Required column '{col}' not found in {file}")
                
                # Convert Date to datetime and set as index
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Ensure numeric values for OHLC
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort by date
                df.sort_index(inplace=True)
                
                # Store in data dictionary
                self.data[pair_name] = df
                logger.info(f"Successfully loaded {pair_name} data with {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                
        return self.data
    
    def validate_data(self) -> Dict[str, Dict]:
        """
        Perform validation checks on loaded data.
        
        Returns:
            Validation report dictionary
        """
        if not self.data:
            logger.warning("No data loaded to validate")
            return {}
        
        for pair_name, df in self.data.items():
            pair_report = {
                'missing_values': {},
                'timestamp_issues': [],
                'ohlc_integrity': [],
                'summary': {}
            }
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                pair_report['missing_values'] = missing[missing > 0].to_dict()
                
            # Check timestamp consistency (look for gaps)
            expected_freq = '1D'  # Assuming daily data
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
            missing_dates = set(date_range) - set(df.index)
            if missing_dates:
                pair_report['timestamp_issues'] = sorted(list(missing_dates))[:10]  # Limit to first 10 issues
                
            # Check OHLC integrity
            integrity_issues = []
            
            # High should be >= Open and Close
            high_open_issues = df[df['High'] < df['Open']].index.tolist()
            high_close_issues = df[df['High'] < df['Close']].index.tolist()
            
            # Low should be <= Open and Close
            low_open_issues = df[df['Low'] > df['Open']].index.tolist()
            low_close_issues = df[df['Low'] > df['Close']].index.tolist()
            
            if high_open_issues:
                integrity_issues.append(f"High < Open on {len(high_open_issues)} days")
            if high_close_issues:
                integrity_issues.append(f"High < Close on {len(high_close_issues)} days")
            if low_open_issues:
                integrity_issues.append(f"Low > Open on {len(low_open_issues)} days")
            if low_close_issues:
                integrity_issues.append(f"Low > Close on {len(low_close_issues)} days")
                
            pair_report['ohlc_integrity'] = integrity_issues
            
            # Summary stats
            pair_report['summary'] = {
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'trading_days': len(df),
                'missing_days': len(missing_dates),
                'integrity_issues': len(integrity_issues),
                'has_missing_values': missing.sum() > 0
            }
            
            self.validation_report[pair_name] = pair_report
            logger.info(f"Validation completed for {pair_name}")
            
        return self.validation_report
    
    def fix_validation_issues(self, interpolate_missing: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fix common validation issues in the data.
        
        Args:
            interpolate_missing: Whether to interpolate missing values
            
        Returns:
            Dictionary of fixed DataFrames
        """
        if not self.validation_report:
            self.validate_data()
            
        for pair_name, report in self.validation_report.items():
            df = self.data[pair_name].copy()
            
            # Fix missing dates
            if report['timestamp_issues']:
                date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1D')
                df = df.reindex(date_range)
                logger.info(f"Reindexed {pair_name} to include {len(report['timestamp_issues'])} missing dates")
            
            # Fix missing values
            if interpolate_missing and report['missing_values']:
                # Use linear interpolation for missing values
                df.interpolate(method='linear', inplace=True)
                # Forward fill any remaining NAs (at the beginning)
                df.fillna(method='ffill', inplace=True)
                # Backward fill any remaining NAs (at the end)
                df.fillna(method='bfill', inplace=True)
                logger.info(f"Interpolated missing values for {pair_name}")
            
            # Fix OHLC integrity issues (if any exist)
            if report['ohlc_integrity']:
                # Ensure High is the highest value
                df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
                # Ensure Low is the lowest value
                df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
                logger.info(f"Fixed OHLC integrity issues for {pair_name}")
            
            # Update the data dictionary
            self.data[pair_name] = df
        
        return self.data
    
    def visualize_data(self, pair_name: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Create visualizations of the forex data.
        
        Args:
            pair_name: Currency pair to visualize
            start_date: Start date for visualization (YYYY-MM-DD)
            end_date: End date for visualization (YYYY-MM-DD)
            save_path: Path to save the visualization, if None, display only
        """
        if pair_name not in self.data:
            logger.error(f"Currency pair '{pair_name}' not found in loaded data")
            return
        
        df = self.data[pair_name].copy()
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) == 0:
            logger.error(f"No data to visualize for the selected date range")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot OHLC data
        df.plot(y=['Open', 'High', 'Low', 'Close'], ax=axs[0], grid=True)
        axs[0].set_title(f'{pair_name} OHLC Data')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        
        # Plot volume (if available) or price changes
        if 'Volume' in df.columns:
            df['Volume'].plot(kind='bar', ax=axs[1], color='blue', alpha=0.5)
            axs[1].set_title('Volume')
            axs[1].set_ylabel('Volume')
        else:
            # Calculate daily price changes as alternative
            df['Price_Change'] = df['Close'].pct_change(fill_method=None) * 100
            # Fill NaN values that might be created by pct_change
            df['Price_Change'].fillna(0, inplace=True)
            df['Price_Change'].plot(kind='bar', ax=axs[1], color=df['Price_Change'].apply(
                lambda x: 'green' if x >= 0 else 'red'))
            axs[1].set_title('Daily Price Change (%)')
            axs[1].set_ylabel('Change (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    def create_candlestick_chart(self, pair_name: str, start_date: Optional[str] = None, 
                               end_date: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Create a candlestick chart for the specified currency pair.
        
        Args:
            pair_name: Currency pair to visualize
            start_date: Start date for visualization (YYYY-MM-DD)
            end_date: End date for visualization (YYYY-MM-DD)
            save_path: Path to save the visualization, if None, display only
        """
        if pair_name not in self.data:
            logger.error(f"Currency pair '{pair_name}' not found in loaded data")
            return
        
        df = self.data[pair_name].copy()
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) == 0:
            logger.error(f"No data to visualize for the selected date range")
            return
        
        # Prepare data for mplfinance
        df = df[['Open', 'High', 'Low', 'Close']]
        
        # Create candlestick chart
        mpf.plot(df, type='candle', style='yahoo', title=f'{pair_name} Candlestick Chart',
                figsize=(12, 8), volume=False, savefig=save_path if save_path else None)
        
        if save_path:
            logger.info(f"Candlestick chart saved to {save_path}")
            
    def get_data_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of loaded data.
        
        Returns:
            Dictionary with summary information for each pair
        """
        summary = {}
        
        for pair_name, df in self.data.items():
            summary[pair_name] = {
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'trading_days': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'descriptive_stats': df.describe().to_dict()
            }
            
        return summary