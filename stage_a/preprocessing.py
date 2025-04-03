"""
Advanced Preprocessing Pipeline for the VMETHOD framework.

This module implements sophisticated time series preprocessing techniques:
- Temporal alignment
- Fractional differentiation
- Market regime segmentation
- Wavelet transform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import pywt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Preprocessing')

class AdvancedPreprocessingPipeline:
    """
    Advanced Preprocessing Pipeline for forex time series data.
    
    Implements sophisticated preprocessing techniques for financial time series:
    - Temporal alignment
    - Fractional differentiation 
    - Market regime segmentation
    - Wavelet transformation
    """
    
    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.preprocessed_data = {}
        
    def _calculate_weights(self, d: float, threshold: float = 1e-5, k_max: int = 500) -> np.ndarray:
        """
        Calculate weights for fractional differentiation.
        
        Args:
            d: Fractional differentiation order (0 <= d <= 1)
            threshold: Weight threshold for cutoff
            k_max: Maximum number of terms to consider
            
        Returns:
            Array of weights
        """
        w = [1.]  # First weight is always 1
        for k in range(1, k_max):
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
        return np.array(w)
    
    def align_time_series(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Synchronize timestamps across all currency pairs.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            
        Returns:
            Dictionary of temporally aligned DataFrames
        """
        if not data_dict:
            logger.warning("No data provided for alignment")
            return {}
        
        # Find the common date range for all pairs
        min_dates = []
        max_dates = []
        
        for pair_name, df in data_dict.items():
            min_dates.append(df.index.min())
            max_dates.append(df.index.max())
        
        common_start = max(min_dates)
        common_end = min(max_dates)
        
        # Create a common date range
        common_range = pd.date_range(common_start, common_end, freq='1D')
        
        aligned_data = {}
        for pair_name, df in data_dict.items():
            # Reindex each DataFrame to the common range
            aligned_df = df.reindex(common_range)
            
            # Interpolate missing values
            aligned_df.interpolate(method='linear', inplace=True)
            aligned_df.ffill(inplace=True)  # Forward fill any remaining NAs
            aligned_df.bfill(inplace=True)  # Backward fill any remaining NAs
            
            aligned_data[pair_name] = aligned_df
            logger.info(f"Aligned {pair_name} data to common date range ({common_start} to {common_end})")
        
        return aligned_data
    
    def apply_fractional_differentiation(self, df: pd.DataFrame, column: str = 'Close', 
                                       d_range: Tuple[float, float, float] = (0.1, 1.0, 0.1)) -> Tuple[pd.DataFrame, float]:
        """
        Apply fractional differentiation to a time series column.
        
        Args:
            df: DataFrame containing time series data
            column: Column name to differentiate
            d_range: Tuple of (start, end, step) for differentiation order search
            
        Returns:
            DataFrame with differentiated series added as a new column, and optimal d value
        """
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            return df, None
        
        series = df[column].values
        n = len(series)
        result_df = df.copy()
        
        # Search for optimal differentiation order
        d_values = np.arange(d_range[0], d_range[1] + d_range[2], d_range[2])
        best_d = None
        best_adf = float('inf')  # Looking for most negative ADF statistic
        
        for d in d_values:
            # Calculate weights for this d
            w = self._calculate_weights(d)
            width = len(w)
            
            # Apply weights to calculate fractionally differentiated series
            frac_diff = np.zeros(n)
            frac_diff[:width] = np.nan  # Initial values can't be calculated
            
            # Compute the differentiated series
            for i in range(width, n):
                # Dot product of weights with the corresponding window of the series
                frac_diff[i] = np.sum(w * series[i - width:i][::-1])
            
            # Create a temp series without NaN values for testing
            temp_series = pd.Series(frac_diff).dropna()
            
            if len(temp_series) <= 10:  # Need enough data for the test
                continue
                
            # Test for stationarity using ADF test
            adf_result = adfuller(temp_series, regression='c', autolag='AIC')
            
            # If this d value gives a more stationary series (more negative ADF statistic)
            if adf_result[0] < best_adf:
                best_adf = adf_result[0]
                best_d = d
        
        if best_d is None:
            logger.warning("Could not find optimal fractional differentiation order")
            return result_df, None
        
        # Apply the best differentiation order
        w = self._calculate_weights(best_d)
        width = len(w)
        
        frac_diff = np.zeros(n)
        frac_diff[:width] = np.nan
        
        # Compute the differentiated series with the optimal d
        for i in range(width, n):
            frac_diff[i] = np.sum(w * series[i - width:i][::-1])
        
        # Add the differentiated series to the result DataFrame
        result_df[f'{column}_frac_diff'] = frac_diff
        result_df[f'{column}_frac_diff_d'] = best_d
        
        logger.info(f"Applied fractional differentiation with d = {best_d:.2f}")
        
        return result_df, best_d
    
    def detect_market_regimes(self, df: pd.DataFrame, column: str = 'Close', 
                            window: int = 20) -> pd.DataFrame:
        """
        Identify market regimes (uptrend, downtrend, sideways).
        
        Args:
            df: DataFrame containing time series data
            column: Column name to analyze
            window: Window size for moving averages
            
        Returns:
            DataFrame with regime classification added
        """
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            return df
        
        result_df = df.copy()
        
        # Calculate simple moving averages
        result_df[f'{column}_SMA'] = result_df[column].rolling(window=window).mean()
        
        # Calculate price slope (rate of change)
        result_df[f'{column}_slope'] = result_df[f'{column}_SMA'].pct_change(periods=window) * 100
        
        # Calculate volatility (standard deviation)
        result_df[f'{column}_volatility'] = result_df[column].rolling(window=window).std()
        
        # Define regimes
        # Uptrend: Positive slope above threshold
        # Downtrend: Negative slope below threshold
        # Sideways: Slope near zero or high volatility with low slope
        
        slope_threshold = 0.5  # 0.5% change over the window period
        
        conditions = [
            (result_df[f'{column}_slope'] > slope_threshold),  # Uptrend
            (result_df[f'{column}_slope'] < -slope_threshold),  # Downtrend
            (True)  # Sideways (default if neither up nor down)
        ]
        
        regime_values = [1, -1, 0]  # 1 for uptrend, -1 for downtrend, 0 for sideways
        regime_labels = ['uptrend', 'downtrend', 'sideways']
        
        result_df[f'{column}_regime'] = np.select(conditions, regime_values)
        result_df[f'{column}_regime_label'] = np.select(conditions, regime_labels)
        
        logger.info(f"Detected market regimes with {window}-day window")
        
        return result_df
    
    def apply_wavelet_transform(self, df: pd.DataFrame, column: str = 'Close', 
                               wavelet: str = 'db8', level: int = 3) -> pd.DataFrame:
        """
        Apply wavelet transformation to decompose price signal.
        
        Args:
            df: DataFrame containing time series data
            column: Column name to analyze
            wavelet: Wavelet type to use
            level: Decomposition level
            
        Returns:
            DataFrame with wavelet components added
        """
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            return df
        
        # Need to handle NaN values before wavelet transform
        series = df[column].ffill().values
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(series, wavelet, level=level)
        
        # Reconstruct components
        result_df = df.copy()
        
        # First coefficient is the approximation (trend)
        result_df[f'{column}_trend'] = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(series))
        
        # Remaining coefficients are details (cycles and noise)
        detail_sum = np.zeros(len(series))
        for i in range(1, len(coeffs)):
            detail = pywt.upcoef('d', coeffs[i], wavelet, level=level-i+1, take=len(series))
            detail_sum += detail
            
            # Add individual detail levels for more granular analysis
            if i <= 3:  # Only store the first few detail levels individually
                result_df[f'{column}_detail_{i}'] = detail
        
        result_df[f'{column}_details'] = detail_sum
        
        logger.info(f"Applied wavelet transform with {wavelet} wavelet at level {level}")
        
        return result_df
    
    def process_pair(self, df: pd.DataFrame, column: str = 'Close', 
                    apply_fractional_diff: bool = True,
                    apply_regime_detection: bool = True,
                    apply_wavelet: bool = True) -> pd.DataFrame:
        """
        Apply the full preprocessing pipeline to a single currency pair.
        
        Args:
            df: DataFrame for a currency pair
            column: Main column to analyze (typically 'Close')
            apply_fractional_diff: Whether to apply fractional differentiation
            apply_regime_detection: Whether to detect market regimes
            apply_wavelet: Whether to apply wavelet transformation
            
        Returns:
            Fully preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # Apply fractional differentiation
        if apply_fractional_diff:
            processed_df, best_d = self.apply_fractional_differentiation(processed_df, column)
            
        # Detect market regimes
        if apply_regime_detection:
            processed_df = self.detect_market_regimes(processed_df, column)
            
        # Apply wavelet transformation
        if apply_wavelet:
            processed_df = self.apply_wavelet_transform(processed_df, column)
            
        return processed_df
    
    def process_all_pairs(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply the full preprocessing pipeline to all currency pairs.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        # First align all time series
        aligned_data = self.align_time_series(data_dict)
        
        # Then apply preprocessing to each pair
        for pair_name, df in aligned_data.items():
            logger.info(f"Processing {pair_name}...")
            self.preprocessed_data[pair_name] = self.process_pair(df)
            
        return self.preprocessed_data
    
    def visualize_preprocessing(self, pair_name: str, column: str = 'Close', 
                               save_path: Optional[str] = None) -> None:
        """
        Create visualizations of preprocessing results.
        
        Args:
            pair_name: Currency pair to visualize
            column: Main column that was analyzed
            save_path: Path to save the visualization, if None, display only
        """
        if pair_name not in self.preprocessed_data:
            logger.error(f"Preprocessed data for '{pair_name}' not found")
            return
            
        df = self.preprocessed_data[pair_name]
        
        # Create a 2x2 grid of plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original vs. Fractionally Differentiated
        if f'{column}_frac_diff' in df.columns:
            ax = axs[0, 0]
            df[[column, f'{column}_frac_diff']].plot(ax=ax)
            ax.set_title(f'Original vs. Fractionally Differentiated ({column})')
            ax.set_ylabel('Price')
            ax.legend()
            
            # Also add ADF test results as text
            original_adf = adfuller(df[column].dropna(), regression='c', autolag='AIC')
            diff_adf = adfuller(df[f'{column}_frac_diff'].dropna(), regression='c', autolag='AIC')
            
            adf_text = (f"ADF p-values:\n"
                       f"Original: {original_adf[1]:.4f}\n"
                       f"Frac. Diff: {diff_adf[1]:.4f}")
            ax.text(0.05, 0.95, adf_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
        
        # Market Regimes
        if f'{column}_regime' in df.columns:
            ax = axs[0, 1]
            
            # Plot price
            ax.plot(df.index, df[column], label=column, color='blue')
            
            # Color the background according to regime
            for regime, color in zip([-1, 0, 1], ['red', 'gray', 'green']):
                mask = df[f'{column}_regime'] == regime
                if mask.any():
                    ax.fill_between(df.index, df[column].min(), df[column].max(), 
                                   where=mask, color=color, alpha=0.2)
            
            ax.set_title(f'Market Regimes ({column})')
            ax.set_ylabel('Price')
            ax.legend()
            
            # Add regime legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.2, label='Uptrend'),
                Patch(facecolor='gray', alpha=0.2, label='Sideways'),
                Patch(facecolor='red', alpha=0.2, label='Downtrend')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
        
        # Wavelet Decomposition
        if f'{column}_trend' in df.columns:
            ax = axs[1, 0]
            df[[column, f'{column}_trend']].plot(ax=ax)
            ax.set_title(f'Original vs. Trend Component ({column})')
            ax.set_ylabel('Price')
            ax.legend()
            
            ax = axs[1, 1]
            if f'{column}_detail_1' in df.columns and f'{column}_detail_2' in df.columns:
                df[[f'{column}_detail_1', f'{column}_detail_2']].plot(ax=ax)
                ax.set_title(f'Wavelet Detail Components ({column})')
                ax.set_ylabel('Magnitude')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Preprocessing visualization saved to {save_path}")
        else:
            plt.show()