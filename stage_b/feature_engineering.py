"""
Enhanced Feature Library for the VMETHOD framework.

This module implements technical indicators for forex time series data:
- Trend indicators: Moving averages, MACD, DMI
- Momentum indicators: RSI, Stochastic, ROC, CCI
- Volatility indicators: Bollinger Bands, ATR, Standard Deviation, Keltner Channels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FeatureEngineering')

class TechnicalIndicator:
    """Base class for all technical indicators."""
    
    def __init__(self, name: str):
        """
        Initialize technical indicator.
        
        Args:
            name: Name of the indicator
        """
        self.name = name
        
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate the indicator values.
        
        Args:
            df: DataFrame containing price data
            **kwargs: Additional parameters for the indicator
            
        Returns:
            DataFrame with indicator values
        """
        raise NotImplementedError("Subclasses must implement calculate method")
        
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close', 
                 save_path: Optional[str] = None) -> None:
        """
        Visualize the indicator alongside price data.
        
        Args:
            df: DataFrame containing price and indicator data
            price_col: Column name for price data
            save_path: Path to save visualization
        """
        raise NotImplementedError("Subclasses must implement visualize method")

class MovingAverage(TechnicalIndicator):
    """Simple and Exponential Moving Averages."""
    
    def __init__(self, name: str = "MovingAverage"):
        """Initialize MovingAverage indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, column: str = 'Close', 
                windows: List[int] = [5, 10, 20, 50, 200], 
                exp: bool = False) -> pd.DataFrame:
        """
        Calculate Simple or Exponential Moving Averages.
        
        Args:
            df: DataFrame containing price data
            column: Column to calculate MA for
            windows: List of window sizes
            exp: If True, calculate EMA instead of SMA
            
        Returns:
            DataFrame with MA values
        """
        result = df.copy()
        
        for window in windows:
            if exp:
                result[f'{column}_EMA_{window}'] = result[column].ewm(span=window, adjust=False).mean()
                logger.info(f"Calculated EMA-{window} for {column}")
            else:
                result[f'{column}_SMA_{window}'] = result[column].rolling(window=window).mean()
                logger.info(f"Calculated SMA-{window} for {column}")
                
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close', 
                ma_cols: Optional[List[str]] = None,
                save_path: Optional[str] = None) -> None:
        """
        Visualize Moving Averages alongside price data.
        
        Args:
            df: DataFrame containing price and MA data
            price_col: Column name for price data
            ma_cols: List of MA column names to plot, if None plot all
            save_path: Path to save visualization
        """
        if ma_cols is None:
            # Find all MA columns
            ma_cols = [col for col in df.columns if ('_SMA_' in col or '_EMA_' in col)]
            
        if not ma_cols:
            logger.warning("No MA columns found for visualization")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot price
        plt.plot(df.index, df[price_col], label=price_col, linewidth=1.5)
        
        # Plot each MA
        for col in ma_cols:
            plt.plot(df.index, df[col], label=col, linewidth=1)
            
        plt.title(f'Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Moving Averages visualization saved to {save_path}")
        else:
            plt.show()


class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence."""
    
    def __init__(self, name: str = "MACD"):
        """Initialize MACD indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, column: str = 'Close', 
                fast_period: int = 12, slow_period: int = 26,
                signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD indicator.
        
        Args:
            df: DataFrame containing price data
            column: Column to calculate MACD for
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD values
        """
        result = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = result[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result[column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        result[f'{column}_MACD'] = fast_ema - slow_ema
        
        # Calculate signal line
        result[f'{column}_MACD_Signal'] = result[f'{column}_MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        result[f'{column}_MACD_Hist'] = result[f'{column}_MACD'] - result[f'{column}_MACD_Signal']
        
        logger.info(f"Calculated MACD for {column}")
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close', 
                save_path: Optional[str] = None) -> None:
        """
        Visualize MACD indicator.
        
        Args:
            df: DataFrame containing price and MACD data
            price_col: Column name for price data
            save_path: Path to save visualization
        """
        macd_col = f'{price_col}_MACD'
        signal_col = f'{price_col}_MACD_Signal'
        hist_col = f'{price_col}_MACD_Hist'
        
        if not all(col in df.columns for col in [macd_col, signal_col, hist_col]):
            logger.warning("MACD columns not found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MACD on bottom subplot
        ax2.plot(df.index, df[macd_col], label='MACD', color='blue')
        ax2.plot(df.index, df[signal_col], label='Signal', color='red')
        
        # Plot histogram as bar chart
        positive_hist = df[hist_col].copy()
        negative_hist = df[hist_col].copy()
        positive_hist[positive_hist <= 0] = np.nan
        negative_hist[negative_hist > 0] = np.nan
        
        ax2.bar(df.index, positive_hist, color='green', label='Positive Histogram', width=1, alpha=0.5)
        ax2.bar(df.index, negative_hist, color='red', label='Negative Histogram', width=1, alpha=0.5)
        
        ax2.set_title('MACD')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"MACD visualization saved to {save_path}")
        else:
            plt.show()
            

class DirectionalMovementIndex(TechnicalIndicator):
    """Directional Movement Index (DMI)."""
    
    def __init__(self, name: str = "DMI"):
        """Initialize DMI indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate DMI indicator.
        
        Args:
            df: DataFrame containing OHLC data
            period: Calculation period
            
        Returns:
            DataFrame with DMI values
        """
        result = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in result.columns for col in required_cols):
            logger.error("Required OHLC columns not found for DMI calculation")
            return result
            
        # Calculate True Range (TR)
        result['TR'] = np.maximum(
            result['High'] - result['Low'],
            np.maximum(
                np.abs(result['High'] - result['Close'].shift(1)),
                np.abs(result['Low'] - result['Close'].shift(1))
            )
        )
        
        # Calculate Directional Movement
        result['Plus_DM'] = np.where(
            (result['High'] - result['High'].shift(1)) > (result['Low'].shift(1) - result['Low']),
            np.maximum(result['High'] - result['High'].shift(1), 0),
            0
        )
        
        result['Minus_DM'] = np.where(
            (result['Low'].shift(1) - result['Low']) > (result['High'] - result['High'].shift(1)),
            np.maximum(result['Low'].shift(1) - result['Low'], 0),
            0
        )
        
        # Calculate smoothed values
        result['ATR'] = result['TR'].rolling(window=period).mean()
        result['Plus_DI'] = 100 * (result['Plus_DM'].rolling(window=period).mean() / result['ATR'])
        result['Minus_DI'] = 100 * (result['Minus_DM'].rolling(window=period).mean() / result['ATR'])
        
        # Calculate Directional Index (DX)
        result['DX'] = 100 * np.abs(result['Plus_DI'] - result['Minus_DI']) / (result['Plus_DI'] + result['Minus_DI'])
        
        # Calculate Average Directional Index (ADX)
        result['ADX'] = result['DX'].rolling(window=period).mean()
        
        # Clean up intermediate columns
        result.drop(['TR', 'Plus_DM', 'Minus_DM'], axis=1, inplace=True)
        
        logger.info(f"Calculated DMI with period {period}")
        return result
    
    def visualize(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Visualize DMI indicator.
        
        Args:
            df: DataFrame containing price and DMI data
            save_path: Path to save visualization
        """
        required_cols = ['Plus_DI', 'Minus_DI', 'ADX']
        if not all(col in df.columns for col in required_cols):
            logger.warning("DMI columns not found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df['Close'], label='Close')
        ax1.set_title('Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot DMI on bottom subplot
        ax2.plot(df.index, df['Plus_DI'], label='+DI', color='green')
        ax2.plot(df.index, df['Minus_DI'], label='-DI', color='red')
        ax2.plot(df.index, df['ADX'], label='ADX', color='blue', linewidth=1.5)
        
        # Add horizontal lines for reference
        ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.3)
        ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_title('Directional Movement Index (DMI)')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"DMI visualization saved to {save_path}")
        else:
            plt.show()


class RSI(TechnicalIndicator):
    """Relative Strength Index."""
    
    def __init__(self, name: str = "RSI"):
        """Initialize RSI indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, column: str = 'Close', 
                periods: List[int] = [14]) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame containing price data
            column: Column to calculate RSI for
            periods: List of periods to calculate RSI
            
        Returns:
            DataFrame with RSI values
        """
        result = df.copy()
        
        for period in periods:
            delta = result[column].diff()
            
            # Calculate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss  # Make losses positive
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            result[f'{column}_RSI_{period}'] = 100 - (100 / (1 + rs))
            
            logger.info(f"Calculated RSI-{period} for {column}")
            
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close',
                 rsi_cols: Optional[List[str]] = None,
                 save_path: Optional[str] = None) -> None:
        """
        Visualize RSI indicator.
        
        Args:
            df: DataFrame containing price and RSI data
            price_col: Column name for price data
            rsi_cols: List of RSI column names to plot, if None plot all
            save_path: Path to save visualization
        """
        if rsi_cols is None:
            # Find all RSI columns
            rsi_cols = [col for col in df.columns if '_RSI_' in col]
            
        if not rsi_cols:
            logger.warning("No RSI columns found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot RSI on bottom subplot
        for col in rsi_cols:
            ax2.plot(df.index, df[col], label=col)
            
        # Add overbought/oversold lines
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_ylabel('RSI Value')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"RSI visualization saved to {save_path}")
        else:
            plt.show()


class Stochastic(TechnicalIndicator):
    """Stochastic Oscillator."""
    
    def __init__(self, name: str = "Stochastic"):
        """Initialize Stochastic indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, k_period: int = 14, 
                d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame containing OHLC data
            k_period: K period
            d_period: D period
            slowing: Slowing period
            
        Returns:
            DataFrame with Stochastic values
        """
        result = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in result.columns for col in required_cols):
            logger.error("Required OHLC columns not found for Stochastic calculation")
            return result
            
        # Calculate %K
        lowest_low = result['Low'].rolling(window=k_period).min()
        highest_high = result['High'].rolling(window=k_period).max()
        
        result['%K_Fast'] = 100 * ((result['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Apply slowing period
        result['%K'] = result['%K_Fast'].rolling(window=slowing).mean()
        
        # Calculate %D
        result['%D'] = result['%K'].rolling(window=d_period).mean()
        
        # Clean up intermediate columns
        result.drop(['%K_Fast'], axis=1, inplace=True)
        
        logger.info(f"Calculated Stochastic Oscillator with K={k_period}, D={d_period}, slowing={slowing}")
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close', 
                save_path: Optional[str] = None) -> None:
        """
        Visualize Stochastic Oscillator.
        
        Args:
            df: DataFrame containing price and Stochastic data
            price_col: Column name for price data
            save_path: Path to save visualization
        """
        if not all(col in df.columns for col in ['%K', '%D']):
            logger.warning("Stochastic columns not found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Stochastic on bottom subplot
        ax2.plot(df.index, df['%K'], label='%K', color='blue')
        ax2.plot(df.index, df['%D'], label='%D', color='red')
        
        # Add overbought/oversold lines
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        
        ax2.set_title('Stochastic Oscillator')
        ax2.set_ylabel('Value')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stochastic Oscillator visualization saved to {save_path}")
        else:
            plt.show()


class ROC(TechnicalIndicator):
    """Rate of Change."""
    
    def __init__(self, name: str = "ROC"):
        """Initialize ROC indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, column: str = 'Close', 
                periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        Calculate Rate of Change.
        
        Args:
            df: DataFrame containing price data
            column: Column to calculate ROC for
            periods: List of periods for ROC calculation
            
        Returns:
            DataFrame with ROC values
        """
        result = df.copy()
        
        for period in periods:
            result[f'{column}_ROC_{period}'] = ((result[column] / result[column].shift(period)) - 1) * 100
            logger.info(f"Calculated ROC-{period} for {column}")
            
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close',
                roc_cols: Optional[List[str]] = None,
                save_path: Optional[str] = None) -> None:
        """
        Visualize ROC indicator.
        
        Args:
            df: DataFrame containing price and ROC data
            price_col: Column name for price data
            roc_cols: List of ROC column names to plot, if None plot all
            save_path: Path to save visualization
        """
        if roc_cols is None:
            # Find all ROC columns
            roc_cols = [col for col in df.columns if '_ROC_' in col]
            
        if not roc_cols:
            logger.warning("No ROC columns found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot ROC on bottom subplot
        for col in roc_cols:
            ax2.plot(df.index, df[col], label=col)
            
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('Rate of Change (ROC)')
        ax2.set_ylabel('Percent (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC visualization saved to {save_path}")
        else:
            plt.show()


class CCI(TechnicalIndicator):
    """Commodity Channel Index."""
    
    def __init__(self, name: str = "CCI"):
        """Initialize CCI indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, periods: List[int] = [20]) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index.
        
        Args:
            df: DataFrame containing OHLC data
            periods: List of periods for CCI calculation
            
        Returns:
            DataFrame with CCI values
        """
        result = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in result.columns for col in required_cols):
            logger.error("Required OHLC columns not found for CCI calculation")
            return result
        
        for period in periods:
            # Calculate Typical Price
            result['TP'] = (result['High'] + result['Low'] + result['Close']) / 3
            
            # Calculate SMA of Typical Price
            result['SMA_TP'] = result['TP'].rolling(window=period).mean()
            
            # Calculate Mean Deviation
            result['MD'] = result['TP'].rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            
            # Calculate CCI
            result[f'CCI_{period}'] = (result['TP'] - result['SMA_TP']) / (0.015 * result['MD'])
            
            # Clean up intermediate columns
            result.drop(['TP', 'SMA_TP', 'MD'], axis=1, inplace=True)
            
            logger.info(f"Calculated CCI-{period}")
            
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close',
                cci_cols: Optional[List[str]] = None,
                save_path: Optional[str] = None) -> None:
        """
        Visualize CCI indicator.
        
        Args:
            df: DataFrame containing price and CCI data
            price_col: Column name for price data
            cci_cols: List of CCI column names to plot, if None plot all
            save_path: Path to save visualization
        """
        if cci_cols is None:
            # Find all CCI columns
            cci_cols = [col for col in df.columns if 'CCI_' in col]
            
        if not cci_cols:
            logger.warning("No CCI columns found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot CCI on bottom subplot
        for col in cci_cols:
            ax2.plot(df.index, df[col], label=col)
            
        # Add overbought/oversold lines
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=-100, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('Commodity Channel Index (CCI)')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CCI visualization saved to {save_path}")
        else:
            plt.show()


class BollingerBands(TechnicalIndicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, name: str = "BollingerBands"):
        """Initialize Bollinger Bands indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, column: str = 'Close', 
                period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame containing price data
            column: Column to calculate bands for
            period: Moving average period
            std_dev: Number of standard deviations for bands
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        result = df.copy()
        
        # Calculate rolling mean and standard deviation
        result[f'{column}_BB_MA'] = result[column].rolling(window=period).mean()
        result[f'{column}_BB_STD'] = result[column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result[f'{column}_BB_Upper'] = result[f'{column}_BB_MA'] + (std_dev * result[f'{column}_BB_STD'])
        result[f'{column}_BB_Lower'] = result[f'{column}_BB_MA'] - (std_dev * result[f'{column}_BB_STD'])
        
        # Calculate bandwidth and %B
        result[f'{column}_BB_Width'] = (result[f'{column}_BB_Upper'] - result[f'{column}_BB_Lower']) / result[f'{column}_BB_MA']
        result[f'{column}_BB_PctB'] = (result[column] - result[f'{column}_BB_Lower']) / (result[f'{column}_BB_Upper'] - result[f'{column}_BB_Lower'])
        
        logger.info(f"Calculated Bollinger Bands with period {period} and {std_dev} standard deviations")
        return result
    
    def visualize(self, df: pd.DataFrame, column: str = 'Close', 
                save_path: Optional[str] = None) -> None:
        """
        Visualize Bollinger Bands.
        
        Args:
            df: DataFrame containing price and Bollinger Bands data
            column: Column name for price data
            save_path: Path to save visualization
        """
        ma_col = f'{column}_BB_MA'
        upper_col = f'{column}_BB_Upper'
        lower_col = f'{column}_BB_Lower'
        width_col = f'{column}_BB_Width'
        
        if not all(col in df.columns for col in [ma_col, upper_col, lower_col, width_col]):
            logger.warning("Bollinger Bands columns not found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and bands on top subplot
        ax1.plot(df.index, df[column], label=column, color='blue', alpha=0.7)
        ax1.plot(df.index, df[ma_col], label='Middle Band', color='black', linestyle='-')
        ax1.plot(df.index, df[upper_col], label='Upper Band', color='red', linestyle='--')
        ax1.plot(df.index, df[lower_col], label='Lower Band', color='green', linestyle='--')
        
        # Fill the area between bands
        ax1.fill_between(df.index, df[upper_col], df[lower_col], color='gray', alpha=0.1)
        
        ax1.set_title('Bollinger Bands')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot bandwidth on bottom subplot
        ax2.plot(df.index, df[width_col], label='Bandwidth', color='purple')
        
        ax2.set_title('Bollinger Bandwidth')
        ax2.set_ylabel('Bandwidth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bollinger Bands visualization saved to {save_path}")
        else:
            plt.show()


class ATR(TechnicalIndicator):
    """Average True Range."""
    
    def __init__(self, name: str = "ATR"):
        """Initialize ATR indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame containing OHLC data
            periods: List of periods for ATR calculation
            
        Returns:
            DataFrame with ATR values
        """
        result = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in result.columns for col in required_cols):
            logger.error("Required OHLC columns not found for ATR calculation")
            return result
            
        # Calculate True Range
        result['TR'] = np.maximum(
            result['High'] - result['Low'],
            np.maximum(
                np.abs(result['High'] - result['Close'].shift(1)),
                np.abs(result['Low'] - result['Close'].shift(1))
            )
        )
        
        # Calculate ATR for each period
        for period in periods:
            if period == 1:
                result[f'ATR_{period}'] = result['TR']
            else:
                # Calculate ATR using Wilder's smoothing method
                atr = np.zeros(len(result))
                atr[:period] = np.nan
                
                # Calculate first ATR
                atr[period-1] = result['TR'].iloc[:period].mean()
                
                # Calculate subsequent ATRs
                for i in range(period, len(result)):
                    atr[i] = ((period - 1) * atr[i-1] + result['TR'].iloc[i]) / period
                    
                result[f'ATR_{period}'] = atr
                
            # Calculate normalized ATR (ATR%)
            result[f'ATR_Pct_{period}'] = (result[f'ATR_{period}'] / result['Close']) * 100
            
            logger.info(f"Calculated ATR-{period}")
            
        # Clean up intermediate columns
        result.drop(['TR'], axis=1, inplace=True)
            
        return result
    
    def visualize(self, df: pd.DataFrame, price_col: str = 'Close',
                atr_cols: Optional[List[str]] = None,
                save_path: Optional[str] = None) -> None:
        """
        Visualize ATR indicator.
        
        Args:
            df: DataFrame containing price and ATR data
            price_col: Column name for price data
            atr_cols: List of ATR column names to plot, if None plot all
            save_path: Path to save visualization
        """
        if atr_cols is None:
            # Find all ATR columns that aren't percent-based
            atr_cols = [col for col in df.columns if 'ATR_' in col and 'ATR_Pct_' not in col]
            
        if not atr_cols:
            logger.warning("No ATR columns found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price on top subplot
        ax1.plot(df.index, df[price_col], label=price_col)
        ax1.set_title(f'{price_col} Price')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot ATR on bottom subplot
        for col in atr_cols:
            ax2.plot(df.index, df[col], label=col)
            
        ax2.set_title('Average True Range (ATR)')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ATR visualization saved to {save_path}")
        else:
            plt.show()


class KeltnerChannels(TechnicalIndicator):
    """Keltner Channels."""
    
    def __init__(self, name: str = "KeltnerChannels"):
        """Initialize Keltner Channels indicator."""
        super().__init__(name)
        
    def calculate(self, df: pd.DataFrame, ema_period: int = 20, 
                atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            df: DataFrame containing OHLC data
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: Multiplier for ATR to set channel width
            
        Returns:
            DataFrame with Keltner Channels values
        """
        result = df.copy()
        
        # Ensure OHLC columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in result.columns for col in required_cols):
            logger.error("Required OHLC columns not found for Keltner Channels calculation")
            return result
            
        # Calculate True Range and ATR if not already calculated
        if f'ATR_{atr_period}' not in result.columns:
            # Calculate True Range
            result['TR'] = np.maximum(
                result['High'] - result['Low'],
                np.maximum(
                    np.abs(result['High'] - result['Close'].shift(1)),
                    np.abs(result['Low'] - result['Close'].shift(1))
                )
            )
            
            # Calculate ATR
            if atr_period == 1:
                result[f'ATR_{atr_period}'] = result['TR']
            else:
                # Calculate ATR using Wilder's smoothing method
                atr = np.zeros(len(result))
                atr[:atr_period] = np.nan
                
                # Calculate first ATR
                atr[atr_period-1] = result['TR'].iloc[:atr_period].mean()
                
                # Calculate subsequent ATRs
                for i in range(atr_period, len(result)):
                    atr[i] = ((atr_period - 1) * atr[i-1] + result['TR'].iloc[i]) / atr_period
                    
                result[f'ATR_{atr_period}'] = atr
                
            # Clean up intermediate columns
            result.drop(['TR'], axis=1, inplace=True)
        
        # Calculate middle line (EMA of Close)
        result['KC_Middle'] = result['Close'].ewm(span=ema_period, adjust=False).mean()
        
        # Calculate upper and lower bands
        result['KC_Upper'] = result['KC_Middle'] + (multiplier * result[f'ATR_{atr_period}'])
        result['KC_Lower'] = result['KC_Middle'] - (multiplier * result[f'ATR_{atr_period}'])
        
        # Calculate width
        result['KC_Width'] = (result['KC_Upper'] - result['KC_Lower']) / result['KC_Middle']
        
        logger.info(f"Calculated Keltner Channels with EMA period {ema_period}, ATR period {atr_period}, and multiplier {multiplier}")
        return result
    
    def visualize(self, df: pd.DataFrame, column: str = 'Close', 
                save_path: Optional[str] = None) -> None:
        """
        Visualize Keltner Channels.
        
        Args:
            df: DataFrame containing price and Keltner Channels data
            column: Column name for price data
            save_path: Path to save visualization
        """
        if not all(col in df.columns for col in ['KC_Middle', 'KC_Upper', 'KC_Lower']):
            logger.warning("Keltner Channels columns not found for visualization")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and channels on top subplot
        ax1.plot(df.index, df[column], label=column, color='blue', alpha=0.7)
        ax1.plot(df.index, df['KC_Middle'], label='Middle Line', color='black', linestyle='-')
        ax1.plot(df.index, df['KC_Upper'], label='Upper Channel', color='red', linestyle='--')
        ax1.plot(df.index, df['KC_Lower'], label='Lower Channel', color='green', linestyle='--')
        
        # Fill the area between channels
        ax1.fill_between(df.index, df['KC_Upper'], df['KC_Lower'], color='gray', alpha=0.1)
        
        ax1.set_title('Keltner Channels')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot width on bottom subplot
        ax2.plot(df.index, df['KC_Width'], label='Channel Width', color='purple')
        
        ax2.set_title('Keltner Channel Width')
        ax2.set_ylabel('Width')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Keltner Channels visualization saved to {save_path}")
        else:
            plt.show()


class EnhancedFeatureLibrary:
    """
    Library of technical indicators for forex time series analysis.
    
    This class provides a comprehensive set of technical indicators:
    - Trend indicators
    - Momentum indicators
    - Volatility indicators
    """
    
    def __init__(self):
        """Initialize the feature library."""
        # Initialize all indicators
        self.moving_average = MovingAverage()
        self.macd = MACD()
        self.dmi = DirectionalMovementIndex()
        self.rsi = RSI()
        self.stochastic = Stochastic()
        self.roc = ROC()
        self.cci = CCI()
        self.bollinger_bands = BollingerBands()
        self.atr = ATR()
        self.keltner_channels = KeltnerChannels()
        
        # Group indicators by type
        self.trend_indicators = [self.moving_average, self.macd, self.dmi]
        self.momentum_indicators = [self.rsi, self.stochastic, self.roc, self.cci]
        self.volatility_indicators = [self.bollinger_bands, self.atr, self.keltner_channels]
        
        self.feature_data = {}  # Store processed feature data
        
    def calculate_all_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate all technical indicators for multiple currency pairs.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            
        Returns:
            Dictionary of DataFrames with calculated indicators
        """
        for pair_name, df in data_dict.items():
            logger.info(f"Calculating features for {pair_name}")
            
            # Start with a copy of the original DataFrame
            result_df = df.copy()
            
            # Calculate trend indicators
            result_df = self.moving_average.calculate(result_df, column='Close', windows=[5, 10, 20, 50, 200], exp=False)
            result_df = self.moving_average.calculate(result_df, column='Close', windows=[5, 10, 20, 50, 200], exp=True)
            result_df = self.macd.calculate(result_df, column='Close')
            result_df = self.dmi.calculate(result_df, period=14)
            
            # Calculate momentum indicators
            result_df = self.rsi.calculate(result_df, column='Close', periods=[7, 14, 21])
            result_df = self.stochastic.calculate(result_df)
            result_df = self.roc.calculate(result_df, column='Close', periods=[5, 10, 20, 60])
            result_df = self.cci.calculate(result_df, periods=[20])
            
            # Calculate volatility indicators
            result_df = self.bollinger_bands.calculate(result_df, column='Close', period=20, std_dev=2.0)
            result_df = self.atr.calculate(result_df, periods=[14])
            result_df = self.keltner_channels.calculate(result_df)
            
            # Store the result
            self.feature_data[pair_name] = result_df
            logger.info(f"Completed feature calculation for {pair_name}")
            
        return self.feature_data
    
    def save_feature_data(self, output_dir: str = 'data/features/') -> None:
        """
        Save feature data to CSV files.
        
        Args:
            output_dir: Directory to save feature data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for pair_name, df in self.feature_data.items():
            output_path = os.path.join(output_dir, f"{pair_name}_features.csv")
            df.to_csv(output_path)
            logger.info(f"Saved feature data for {pair_name} to {output_path}")
    
    def create_feature_visualizations(self, pair_name: str, output_dir: str = 'results/figures/features/') -> None:
        """
        Create visualizations for all indicators.
        
        Args:
            pair_name: Currency pair to visualize
            output_dir: Directory to save visualizations
        """
        if pair_name not in self.feature_data:
            logger.error(f"No feature data found for {pair_name}")
            return
            
        df = self.feature_data[pair_name]
        os.makedirs(output_dir, exist_ok=True)
        
        # Trend indicator visualizations
        self.moving_average.visualize(
            df, 
            price_col='Close',
            ma_cols=['Close_SMA_20', 'Close_SMA_50', 'Close_SMA_200', 'Close_EMA_20', 'Close_EMA_50', 'Close_EMA_200'],
            save_path=os.path.join(output_dir, f"{pair_name}_moving_averages.png")
        )
        
        self.macd.visualize(
            df,
            price_col='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_macd.png")
        )
        
        self.dmi.visualize(
            df,
            save_path=os.path.join(output_dir, f"{pair_name}_dmi.png")
        )
        
        # Momentum indicator visualizations
        self.rsi.visualize(
            df,
            price_col='Close',
            rsi_cols=['Close_RSI_14'],
            save_path=os.path.join(output_dir, f"{pair_name}_rsi.png")
        )
        
        self.stochastic.visualize(
            df,
            price_col='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_stochastic.png")
        )
        
        self.roc.visualize(
            df,
            price_col='Close',
            roc_cols=['Close_ROC_10', 'Close_ROC_20'],
            save_path=os.path.join(output_dir, f"{pair_name}_roc.png")
        )
        
        self.cci.visualize(
            df,
            price_col='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_cci.png")
        )
        
        # Volatility indicator visualizations
        self.bollinger_bands.visualize(
            df,
            column='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_bollinger_bands.png")
        )
        
        self.atr.visualize(
            df,
            price_col='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_atr.png")
        )
        
        self.keltner_channels.visualize(
            df,
            column='Close',
            save_path=os.path.join(output_dir, f"{pair_name}_keltner_channels.png")
        )
        
        logger.info(f"Created feature visualizations for {pair_name}")