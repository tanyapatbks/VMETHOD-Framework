"""
Pattern Recognition System for the VMETHOD framework.

This module implements techniques to identify specific price patterns:
- Candlestick patterns (Doji, Hammer, Engulfing, etc.)
- Technical reversal points (Head and Shoulders, Double Tops/Bottoms, etc.)
- Breakout detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PatternRecognition')

class CandlestickPatterns:
    """
    Candlestick pattern detection for forex data.
    
    Detects common candlestick patterns including:
    - Doji
    - Hammer/Shooting Star
    - Engulfing Patterns
    - Morning/Evening Star
    """
    
    def __init__(self):
        """Initialize candlestick pattern detector."""
        # Dictionary to store pattern detection results
        self.patterns = {}
        
    def _calculate_body_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candle body size and shadow lengths.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added body size metrics
        """
        result = df.copy()
        
        # Calculate absolute body size
        result['body_size'] = np.abs(result['Close'] - result['Open'])
        
        # Calculate relative body size (% of high-low range)
        result['body_size_pct'] = result['body_size'] / (result['High'] - result['Low'])
        
        # Calculate upper and lower shadows
        result['upper_shadow'] = result['High'] - result[['Open', 'Close']].max(axis=1)
        result['lower_shadow'] = result[['Open', 'Close']].min(axis=1) - result['Low']
        
        # Calculate upper and lower shadow percentages
        result['upper_shadow_pct'] = result['upper_shadow'] / (result['High'] - result['Low'])
        result['lower_shadow_pct'] = result['lower_shadow'] / (result['High'] - result['Low'])
        
        # Calculate direction (bullish or bearish)
        result['direction'] = np.where(result['Close'] >= result['Open'], 1, -1)
        
        return result
        
    def detect_doji(self, df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
        """
        Detect Doji candlestick patterns.
        
        Args:
            df: DataFrame with OHLC data
            threshold: Maximum body size relative to high-low range
            
        Returns:
            DataFrame with Doji pattern detection
        """
        result = self._calculate_body_size(df)
        
        # Detect Doji (tiny body)
        result['doji'] = (result['body_size_pct'] <= threshold)
        
        # Different types of Doji
        result['dragonfly_doji'] = (
            result['doji'] & 
            (result['upper_shadow_pct'] <= threshold) & 
            (result['lower_shadow_pct'] >= 0.5)
        )
        
        result['gravestone_doji'] = (
            result['doji'] & 
            (result['upper_shadow_pct'] >= 0.5) & 
            (result['lower_shadow_pct'] <= threshold)
        )
        
        result['long_legged_doji'] = (
            result['doji'] & 
            (result['upper_shadow_pct'] >= 0.2) & 
            (result['lower_shadow_pct'] >= 0.2)
        )
        
        logger.info("Detected Doji patterns")
        return result
        
    def detect_hammer(self, df: pd.DataFrame, body_threshold: float = 0.3, 
                     shadow_threshold: float = 0.7) -> pd.DataFrame:
        """
        Detect Hammer and Shooting Star patterns.
        
        Args:
            df: DataFrame with OHLC and body size data
            body_threshold: Maximum body size relative to high-low range
            shadow_threshold: Minimum shadow size relative to high-low range
            
        Returns:
            DataFrame with Hammer pattern detection
        """
        if 'body_size_pct' not in df.columns:
            result = self._calculate_body_size(df)
        else:
            result = df.copy()
        
        # Detect Hammer (small body, long lower shadow, small upper shadow)
        result['hammer'] = (
            (result['body_size_pct'] <= body_threshold) & 
            (result['lower_shadow_pct'] >= shadow_threshold) &
            (result['upper_shadow_pct'] <= 0.1)
        )
        
        # Detect Shooting Star (small body, long upper shadow, small lower shadow)
        result['shooting_star'] = (
            (result['body_size_pct'] <= body_threshold) & 
            (result['upper_shadow_pct'] >= shadow_threshold) &
            (result['lower_shadow_pct'] <= 0.1)
        )
        
        logger.info("Detected Hammer and Shooting Star patterns")
        return result
        
    def detect_engulfing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Bullish and Bearish Engulfing patterns.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with Engulfing pattern detection
        """
        if 'direction' not in df.columns:
            result = self._calculate_body_size(df)
        else:
            result = df.copy()
            
        # Detect Bullish Engulfing (current candle bullish, previous bearish, current body engulfs previous)
        result['bullish_engulfing'] = (
            (result['direction'] == 1) & 
            (result['direction'].shift(1) == -1) &
            (result['Open'] < result['Close'].shift(1)) &
            (result['Close'] > result['Open'].shift(1))
        )
        
        # Detect Bearish Engulfing (current candle bearish, previous bullish, current body engulfs previous)
        result['bearish_engulfing'] = (
            (result['direction'] == -1) & 
            (result['direction'].shift(1) == 1) &
            (result['Open'] > result['Close'].shift(1)) &
            (result['Close'] < result['Open'].shift(1))
        )
        
        logger.info("Detected Engulfing patterns")
        return result
        
    def detect_stars(self, df: pd.DataFrame, doji_threshold: float = 0.05, 
                    gap_threshold: float = 0.01) -> pd.DataFrame:
        """
        Detect Morning Star and Evening Star patterns.
        
        Args:
            df: DataFrame with OHLC data
            doji_threshold: Maximum body size for middle candle
            gap_threshold: Minimum gap between bodies
            
        Returns:
            DataFrame with Star pattern detection
        """
        if 'body_size_pct' not in df.columns or 'direction' not in df.columns:
            result = self._calculate_body_size(df)
        else:
            result = df.copy()
            
        # Calculate body ranges
        result['body_high'] = result[['Open', 'Close']].max(axis=1)
        result['body_low'] = result[['Open', 'Close']].min(axis=1)
        
        # Detect Morning Star (bearish, small middle, bullish)
        result['morning_star'] = False
        for i in range(2, len(result)):
            first_bearish = result['direction'].iloc[i-2] == -1
            middle_small = result['body_size_pct'].iloc[i-1] <= doji_threshold
            last_bullish = result['direction'].iloc[i] == 1
            
            gap_down = result['body_low'].iloc[i-1] < (result['body_low'].iloc[i-2] - gap_threshold)
            gap_up = result['body_low'].iloc[i] > (result['body_high'].iloc[i-1] + gap_threshold)
            
            if first_bearish and middle_small and last_bullish and gap_down and gap_up:
                result['morning_star'].iloc[i] = True
        
        # Detect Evening Star (bullish, small middle, bearish)
        result['evening_star'] = False
        for i in range(2, len(result)):
            first_bullish = result['direction'].iloc[i-2] == 1
            middle_small = result['body_size_pct'].iloc[i-1] <= doji_threshold
            last_bearish = result['direction'].iloc[i] == -1
            
            gap_up = result['body_high'].iloc[i-1] > (result['body_high'].iloc[i-2] + gap_threshold)
            gap_down = result['body_high'].iloc[i] < (result['body_low'].iloc[i-1] - gap_threshold)
            
            if first_bullish and middle_small and last_bearish and gap_up and gap_down:
                result['evening_star'].iloc[i] = True
        
        logger.info("Detected Morning Star and Evening Star patterns")
        return result
        
    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns in one pass.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all pattern detections
        """
        logger.info("Detecting all candlestick patterns")
        
        # Calculate basic metrics first
        result = self._calculate_body_size(df)
        
        # Detect patterns
        result = self.detect_doji(result)
        result = self.detect_hammer(result)
        result = self.detect_engulfing(result)
        result = self.detect_stars(result)
        
        # Add a summary column for any pattern
        pattern_columns = [
            'doji', 'dragonfly_doji', 'gravestone_doji', 'long_legged_doji',
            'hammer', 'shooting_star', 
            'bullish_engulfing', 'bearish_engulfing',
            'morning_star', 'evening_star'
        ]
        
        result['has_pattern'] = result[pattern_columns].any(axis=1)
        
        # Calculate pattern confidence scores
        # For now, a simple binary score (0 or 1)
        result['pattern_confidence'] = result['has_pattern'].astype(int)
        
        # Clean up intermediate columns if desired
        # result.drop(['body_size', 'body_size_pct', 'upper_shadow', 'lower_shadow', 'body_high', 'body_low'], axis=1, inplace=True)
        
        logger.info("Completed candlestick pattern detection")
        return result
        
    def visualize_patterns(self, df: pd.DataFrame, start_idx: int = None, end_idx: int = None, 
                          save_path: Optional[str] = None) -> None:
        """
        Visualize detected candlestick patterns.
        
        Args:
            df: DataFrame with pattern detections
            start_idx: Start index for visualization
            end_idx: End index for visualization
            save_path: Path to save visualization
        """
        # Check if pattern columns exist
        pattern_columns = [
            'doji', 'hammer', 'shooting_star', 
            'bullish_engulfing', 'bearish_engulfing',
            'morning_star', 'evening_star'
        ]
        
        if not all(col in df.columns for col in pattern_columns + ['Open', 'High', 'Low', 'Close']):
            logger.warning("Required pattern columns not found for visualization")
            return
            
        # Subset data if requested
        if start_idx is not None or end_idx is not None:
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(df)
                
            plot_df = df.iloc[start_idx:end_idx].copy()
        else:
            plot_df = df.copy()
        
        plt.figure(figsize=(14, 10))
        
        # Plot OHLC data as candlesticks
        for i in range(len(plot_df)):
            # Determine if candle is bullish or bearish
            if plot_df['Close'].iloc[i] >= plot_df['Open'].iloc[i]:
                color = 'green'
                body_bottom = plot_df['Open'].iloc[i]
                body_top = plot_df['Close'].iloc[i]
            else:
                color = 'red'
                body_bottom = plot_df['Close'].iloc[i]
                body_top = plot_df['Open'].iloc[i]
                
            # Plot body
            plt.bar(i, body_top - body_bottom, bottom=body_bottom, color=color, width=0.6, alpha=0.6)
            
            # Plot wicks
            plt.plot([i, i], [plot_df['Low'].iloc[i], body_bottom], color='black', linewidth=1)
            plt.plot([i, i], [body_top, plot_df['High'].iloc[i]], color='black', linewidth=1)
            
            # Annotate patterns
            pattern_text = ""
            
            if plot_df['doji'].iloc[i]:
                pattern_text += "D "
            if plot_df['hammer'].iloc[i]:
                pattern_text += "H "
            if plot_df['shooting_star'].iloc[i]:
                pattern_text += "SS "
            if plot_df['bullish_engulfing'].iloc[i]:
                pattern_text += "BE "
            if plot_df['bearish_engulfing'].iloc[i]:
                pattern_text += "BE- "
            if plot_df['morning_star'].iloc[i]:
                pattern_text += "MS "
            if plot_df['evening_star'].iloc[i]:
                pattern_text += "ES "
                
            if pattern_text:
                plt.annotate(pattern_text, (i, plot_df['High'].iloc[i]), 
                             textcoords="offset points", xytext=(0, 5), 
                             ha='center', fontsize=8, color='blue')
        
        plt.title('Candlestick Patterns')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Add legend for pattern abbreviations
        legend_text = "D = Doji, H = Hammer, SS = Shooting Star\nBE = Bullish Engulfing, BE- = Bearish Engulfing\nMS = Morning Star, ES = Evening Star"
        plt.figtext(0.01, 0.01, legend_text, fontsize=10, ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Candlestick pattern visualization saved to {save_path}")
        else:
            plt.show()


class TechnicalReversalPatterns:
    """
    Technical reversal pattern detection.
    
    Detects common reversal patterns including:
    - Head and Shoulders
    - Double Tops/Bottoms
    - Support and Resistance Levels
    - Fibonacci Retracement Levels
    """
    
    def __init__(self):
        """Initialize technical reversal pattern detector."""
        # Dictionary to store pattern detection results
        self.patterns = {}
        
    def detect_peaks_and_valleys(self, df: pd.DataFrame, column: str = 'Close', 
                                prominence: float = 0.01, width: int = 5) -> pd.DataFrame:
        """
        Detect significant peaks and valleys in price data.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            prominence: Minimum peak prominence (relative to price)
            width: Minimum peak width
            
        Returns:
            DataFrame with peak and valley detections
        """
        result = df.copy()
        
        # Calculate min/max prominence threshold 
        price_range = result[column].max() - result[column].min()
        min_prominence = price_range * prominence
        
        # Find peaks (tops)
        peaks, peak_properties = signal.find_peaks(
            result[column].values, 
            prominence=min_prominence, 
            width=width
        )
        
        # Find valleys (bottoms) by inverting the signal
        valleys, valley_properties = signal.find_peaks(
            -result[column].values,
            prominence=min_prominence,
            width=width
        )
        
        # Create indicator columns for peaks and valleys
        result['is_peak'] = False
        result['is_valley'] = False
        result['peak_prominence'] = np.nan
        result['valley_prominence'] = np.nan
        
        # Mark peaks and store prominences
        result.loc[result.index[peaks], 'is_peak'] = True
        result.loc[result.index[peaks], 'peak_prominence'] = peak_properties['prominences']
        
        # Mark valleys and store prominences
        result.loc[result.index[valleys], 'is_valley'] = True
        result.loc[result.index[valleys], 'valley_prominence'] = valley_properties['prominences']
        
        logger.info(f"Detected {len(peaks)} peaks and {len(valleys)} valleys")
        return result
        
    def detect_head_and_shoulders(self, df: pd.DataFrame, column: str = 'Close',
                                 max_width: int = 60) -> pd.DataFrame:
        """
        Detect Head and Shoulders patterns.
        
        Args:
            df: DataFrame with price data and peak/valley detections
            column: Column to analyze
            max_width: Maximum pattern width in bars
            
        Returns:
            DataFrame with Head and Shoulders detections
        """
        if not {'is_peak', 'is_valley'}.issubset(df.columns):
            logger.warning("Peak and valley detections required for Head and Shoulders pattern detection")
            return df
            
        result = df.copy()
        
        # Initialize pattern columns
        result['hs_top'] = False
        result['hs_bottom'] = False
        result['hs_confidence'] = 0.0
        
        # Get indices of peaks and valleys
        peak_indices = np.where(result['is_peak'])[0]
        valley_indices = np.where(result['is_valley'])[0]
        
        # Function to check for head and shoulders top pattern
        def check_hs_top(idx1, idx2, idx3, idx4, idx5):
            # Check if we have enough indices
            if any(idx < 0 or idx >= len(result) for idx in [idx1, idx2, idx3, idx4, idx5]):
                return False, 0.0
                
            # Check pattern width
            if (idx5 - idx1) > max_width:
                return False, 0.0
                
            # Get prices at each point
            left_shoulder = result[column].iloc[idx1]
            left_valley = result[column].iloc[idx2]
            head = result[column].iloc[idx3]
            right_valley = result[column].iloc[idx4]
            right_shoulder = result[column].iloc[idx5]
            
            # Check if head is highest point
            if head <= left_shoulder or head <= right_shoulder:
                return False, 0.0
                
            # Check if shoulders are at similar heights
            shoulder_diff = abs(left_shoulder - right_shoulder)
            price_range = result[column].max() - result[column].min()
            if shoulder_diff > (0.1 * price_range):
                return False, 0.0
                
            # Check if valleys are at similar heights
            valley_diff = abs(left_valley - right_valley)
            if valley_diff > (0.1 * price_range):
                return False, 0.0
                
            # Calculate confidence score based on pattern quality
            confidence = 1.0
            
            # Adjust for symmetry
            left_width = idx3 - idx1
            right_width = idx5 - idx3
            width_ratio = min(left_width, right_width) / max(left_width, right_width)
            confidence *= width_ratio
            
            # Adjust for head prominence
            head_height = head - max(left_valley, right_valley)
            shoulder_height = (left_shoulder + right_shoulder) / 2 - max(left_valley, right_valley)
            if shoulder_height > 0:
                head_ratio = head_height / shoulder_height
                # If head is much higher than shoulders, increase confidence
                confidence *= min(1.0, head_ratio / 2)
            
            return True, confidence
        
        # Detect Head and Shoulders Top pattern
        for i in range(len(peak_indices) - 2):
            for j in range(len(valley_indices) - 1):
                if valley_indices[j] > peak_indices[i] and valley_indices[j] < peak_indices[i+1]:
                    for k in range(j+1, len(valley_indices)):
                        if valley_indices[k] > peak_indices[i+1] and valley_indices[k] < peak_indices[i+2]:
                            
                            # Check the pattern
                            is_hs, confidence = check_hs_top(
                                peak_indices[i], 
                                valley_indices[j], 
                                peak_indices[i+1], 
                                valley_indices[k], 
                                peak_indices[i+2]
                            )
                            
                            if is_hs and confidence > 0:
                                # Mark the head as a head and shoulders pattern
                                idx = peak_indices[i+1]
                                result.loc[result.index[idx], 'hs_top'] = True
                                result.loc[result.index[idx], 'hs_confidence'] = confidence
        
        # Similar logic for inverse head and shoulders (bottom)
        # Just swap peaks and valleys
        def check_hs_bottom(idx1, idx2, idx3, idx4, idx5):
            # Check if we have enough indices
            if any(idx < 0 or idx >= len(result) for idx in [idx1, idx2, idx3, idx4, idx5]):
                return False, 0.0
                
            # Check pattern width
            if (idx5 - idx1) > max_width:
                return False, 0.0
                
            # Get prices at each point
            left_shoulder = result[column].iloc[idx1]
            left_peak = result[column].iloc[idx2]
            head = result[column].iloc[idx3]
            right_peak = result[column].iloc[idx4]
            right_shoulder = result[column].iloc[idx5]
            
            # Check if head is lowest point
            if head >= left_shoulder or head >= right_shoulder:
                return False, 0.0
                
            # Check if shoulders are at similar heights
            shoulder_diff = abs(left_shoulder - right_shoulder)
            price_range = result[column].max() - result[column].min()
            if shoulder_diff > (0.1 * price_range):
                return False, 0.0
                
            # Check if peaks are at similar heights
            peak_diff = abs(left_peak - right_peak)
            if peak_diff > (0.1 * price_range):
                return False, 0.0
                
            # Calculate confidence score based on pattern quality
            confidence = 1.0
            
            # Adjust for symmetry
            left_width = idx3 - idx1
            right_width = idx5 - idx3
            width_ratio = min(left_width, right_width) / max(left_width, right_width)
            confidence *= width_ratio
            
            # Adjust for head prominence
            head_depth = min(left_peak, right_peak) - head
            shoulder_depth = min(left_peak, right_peak) - (left_shoulder + right_shoulder) / 2
            if shoulder_depth > 0:
                head_ratio = head_depth / shoulder_depth
                # If head is much lower than shoulders, increase confidence
                confidence *= min(1.0, head_ratio / 2)
            
            return True, confidence
        
        # Detect Inverse Head and Shoulders (Bottom) pattern
        for i in range(len(valley_indices) - 2):
            for j in range(len(peak_indices) - 1):
                if peak_indices[j] > valley_indices[i] and peak_indices[j] < valley_indices[i+1]:
                    for k in range(j+1, len(peak_indices)):
                        if peak_indices[k] > valley_indices[i+1] and peak_indices[k] < valley_indices[i+2]:
                            
                            # Check the pattern
                            is_hs, confidence = check_hs_bottom(
                                valley_indices[i], 
                                peak_indices[j], 
                                valley_indices[i+1], 
                                peak_indices[k], 
                                valley_indices[i+2]
                            )
                            
                            if is_hs and confidence > 0:
                                # Mark the head as an inverse head and shoulders pattern
                                idx = valley_indices[i+1]
                                result.loc[result.index[idx], 'hs_bottom'] = True
                                result.loc[result.index[idx], 'hs_confidence'] = confidence
        
        logger.info("Detected Head and Shoulders patterns")
        return result
        
    def detect_double_patterns(self, df: pd.DataFrame, column: str = 'Close',
                              max_width: int = 60, max_diff_pct: float = 0.03) -> pd.DataFrame:
        """
        Detect Double Top and Double Bottom patterns.
        
        Args:
            df: DataFrame with price data and peak/valley detections
            column: Column to analyze
            max_width: Maximum pattern width in bars
            max_diff_pct: Maximum difference between tops/bottoms as percent of price
            
        Returns:
            DataFrame with Double Top and Double Bottom detections
        """
        if not {'is_peak', 'is_valley'}.issubset(df.columns):
            logger.warning("Peak and valley detections required for Double Top/Bottom pattern detection")
            return df
            
        result = df.copy()
        
        # Initialize pattern columns
        result['double_top'] = False
        result['double_bottom'] = False
        result['double_confidence'] = 0.0
        
        # Get indices of peaks and valleys
        peak_indices = np.where(result['is_peak'])[0]
        valley_indices = np.where(result['is_valley'])[0]
        
        # Detect Double Top pattern
        for i in range(len(peak_indices) - 1):
            peak1_idx = peak_indices[i]
            peak2_idx = peak_indices[i+1]
            
            # Check if peaks are not too far apart
            if (peak2_idx - peak1_idx) > max_width:
                continue
                
            # Get peak values
            peak1_val = result[column].iloc[peak1_idx]
            peak2_val = result[column].iloc[peak2_idx]
            
            # Calculate price range for this window
            window_min = result[column].iloc[peak1_idx:peak2_idx+1].min()
            price_range = max(peak1_val, peak2_val) - window_min
            
            # Check if peaks are at similar levels
            peaks_diff = abs(peak1_val - peak2_val)
            if peaks_diff > (max_diff_pct * price_range):
                continue
                
            # Look for a valley between the peaks
            between_valleys = [idx for idx in valley_indices if peak1_idx < idx < peak2_idx]
            if not between_valleys:
                continue
                
            # Find the lowest valley between peaks
            valley_idx = between_valleys[np.argmin([result[column].iloc[idx] for idx in between_valleys])]
            valley_val = result[column].iloc[valley_idx]
            
            # Calculate pattern characteristics
            depth = (peak1_val + peak2_val) / 2 - valley_val
            
            # Calculate confidence score
            confidence = 1.0
            
            # Adjust for similarity of peaks
            peak_similarity = 1.0 - (peaks_diff / price_range)
            confidence *= peak_similarity
            
            # Adjust for depth of the pattern
            if price_range > 0:
                depth_ratio = depth / price_range
                confidence *= min(1.0, depth_ratio * 2)
            
            # Mark both peaks as double top
            result.loc[result.index[peak1_idx], 'double_top'] = True
            result.loc[result.index[peak1_idx], 'double_confidence'] = confidence
            result.loc[result.index[peak2_idx], 'double_top'] = True
            result.loc[result.index[peak2_idx], 'double_confidence'] = confidence
        
        # Detect Double Bottom pattern (similar logic but for valleys)
        for i in range(len(valley_indices) - 1):
            valley1_idx = valley_indices[i]
            valley2_idx = valley_indices[i+1]
            
            # Check if valleys are not too far apart
            if (valley2_idx - valley1_idx) > max_width:
                continue
                
            # Get valley values
            valley1_val = result[column].iloc[valley1_idx]
            valley2_val = result[column].iloc[valley2_idx]
            
            # Calculate price range for this window
            window_max = result[column].iloc[valley1_idx:valley2_idx+1].max()
            price_range = window_max - min(valley1_val, valley2_val)
            
            # Check if valleys are at similar levels
            valleys_diff = abs(valley1_val - valley2_val)
            if valleys_diff > (max_diff_pct * price_range):
                continue
                
            # Look for a peak between the valleys
            between_peaks = [idx for idx in peak_indices if valley1_idx < idx < valley2_idx]
            if not between_peaks:
                continue
                
            # Find the highest peak between valleys
            peak_idx = between_peaks[np.argmax([result[column].iloc[idx] for idx in between_peaks])]
            peak_val = result[column].iloc[peak_idx]
            
            # Calculate pattern characteristics
            depth = peak_val - (valley1_val + valley2_val) / 2
            
            # Calculate confidence score
            confidence = 1.0
            
            # Adjust for similarity of valleys
            valley_similarity = 1.0 - (valleys_diff / price_range)
            confidence *= valley_similarity
            
            # Adjust for depth of the pattern
            if price_range > 0:
                depth_ratio = depth / price_range
                confidence *= min(1.0, depth_ratio * 2)
            
            # Mark both valleys as double bottom
            result.loc[result.index[valley1_idx], 'double_bottom'] = True
            result.loc[result.index[valley1_idx], 'double_confidence'] = confidence
            result.loc[result.index[valley2_idx], 'double_bottom'] = True
            result.loc[result.index[valley2_idx], 'double_confidence'] = confidence
        
        logger.info("Detected Double Top and Double Bottom patterns")
        return result
        
    def detect_support_resistance(self, df: pd.DataFrame, column: str = 'Close',
                                 lookback: int = 100, zone_width_pct: float = 0.01, 
                                 touch_count: int = 3) -> pd.DataFrame:
        """
        Detect Support and Resistance levels.
        
        Args:
            df: DataFrame with price data and peak/valley detections
            column: Column to analyze
            lookback: Number of periods to look back
            zone_width_pct: Width of support/resistance zones as percentage of price
            touch_count: Minimum number of touches to qualify as support/resistance
            
        Returns:
            DataFrame with Support and Resistance level detections
        """
        if not {'is_peak', 'is_valley'}.issubset(df.columns):
            logger.warning("Peak and valley detections required for Support/Resistance detection")
            return df
            
        result = df.copy()
        
        # Initialize columns for support and resistance
        result['support_level'] = 0.0
        result['resistance_level'] = 0.0
        result['support_strength'] = 0
        result['resistance_strength'] = 0
        
        # Process each data point
        for i in range(lookback, len(result)):
            # Define current window
            window = result.iloc[i-lookback:i+1]
            
            # Get peaks and valleys in this window
            window_peaks = window[window['is_peak']]
            window_valleys = window[window['is_valley']]
            
            # Calculate zone width
            price_range = window[column].max() - window[column].min()
            zone_width = price_range * zone_width_pct
            
            # Cluster peaks to find resistance zones
            if len(window_peaks) > 0:
                # Sort peaks by price level
                sorted_peaks = window_peaks.sort_values(column)
                
                # Cluster peaks that are close to each other
                clusters = []
                current_cluster = [sorted_peaks.index[0]]
                current_level = sorted_peaks[column].iloc[0]
                
                for j in range(1, len(sorted_peaks)):
                    if abs(sorted_peaks[column].iloc[j] - current_level) <= zone_width:
                        # Add to current cluster
                        current_cluster.append(sorted_peaks.index[j])
                    else:
                        # Start a new cluster
                        clusters.append(current_cluster)
                        current_cluster = [sorted_peaks.index[j]]
                        current_level = sorted_peaks[column].iloc[j]
                
                # Add the last cluster
                clusters.append(current_cluster)
                
                # Find the strongest resistance (most touches)
                strongest_cluster = max(clusters, key=len)
                if len(strongest_cluster) >= touch_count:
                    # Calculate the average level
                    resistance_level = window.loc[strongest_cluster, column].mean()
                    
                    # Record resistance
                    result.loc[result.index[i], 'resistance_level'] = resistance_level
                    result.loc[result.index[i], 'resistance_strength'] = len(strongest_cluster)
            
            # Cluster valleys to find support zones (similar logic)
            if len(window_valleys) > 0:
                # Sort valleys by price level
                sorted_valleys = window_valleys.sort_values(column)
                
                # Cluster valleys that are close to each other
                clusters = []
                current_cluster = [sorted_valleys.index[0]]
                current_level = sorted_valleys[column].iloc[0]
                
                for j in range(1, len(sorted_valleys)):
                    if abs(sorted_valleys[column].iloc[j] - current_level) <= zone_width:
                        # Add to current cluster
                        current_cluster.append(sorted_valleys.index[j])
                    else:
                        # Start a new cluster
                        clusters.append(current_cluster)
                        current_cluster = [sorted_valleys.index[j]]
                        current_level = sorted_valleys[column].iloc[j]
                
                # Add the last cluster
                clusters.append(current_cluster)
                
                # Find the strongest support (most touches)
                strongest_cluster = max(clusters, key=len)
                if len(strongest_cluster) >= touch_count:
                    # Calculate the average level
                    support_level = window.loc[strongest_cluster, column].mean()
                    
                    # Record support
                    result.loc[result.index[i], 'support_level'] = support_level
                    result.loc[result.index[i], 'support_strength'] = len(strongest_cluster)
        
        # Clean up zero values
        result.loc[result['support_level'] == 0, 'support_level'] = np.nan
        result.loc[result['resistance_level'] == 0, 'resistance_level'] = np.nan
        
        logger.info("Detected Support and Resistance levels")
        return result
        
    def calculate_fibonacci_levels(self, df: pd.DataFrame, column: str = 'Close',
                                  lookback: int = 100) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement Levels.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            lookback: Number of periods to look back
            
        Returns:
            DataFrame with Fibonacci level calculations
        """
        result = df.copy()
        
        # Initialize Fibonacci level columns
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        for level in fib_levels:
            result[f'fib_{level:.3f}'] = np.nan
        
        # Process each data point
        for i in range(lookback, len(result)):
            # Define current window
            window = result.iloc[i-lookback:i+1]
            
            # Find the highest high and lowest low in the window
            highest = window[column].max()
            lowest = window[column].min()
            
            # Calculate Fibonacci retracement levels (for both uptrends and downtrends)
            if highest > lowest:
                # For uptrend (retracing down from high)
                range_price = highest - lowest
                for level in fib_levels:
                    result.loc[result.index[i], f'fib_{level:.3f}'] = highest - (range_price * level)
        
        logger.info("Calculated Fibonacci Retracement Levels")
        return result
        
    def detect_all_patterns(self, df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """
        Detect all technical reversal patterns in one pass.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            
        Returns:
            DataFrame with all pattern detections
        """
        logger.info("Detecting all technical reversal patterns")
        
        # Start with peak and valley detection
        result = self.detect_peaks_and_valleys(df, column)
        
        # Detect patterns
        result = self.detect_head_and_shoulders(result, column)
        result = self.detect_double_patterns(result, column)
        result = self.detect_support_resistance(result, column)
        result = self.calculate_fibonacci_levels(result, column)
        
        # Add a summary column for any technical reversal pattern
        pattern_columns = ['hs_top', 'hs_bottom', 'double_top', 'double_bottom']
        
        result['has_reversal_pattern'] = result[pattern_columns].any(axis=1)
        
        logger.info("Completed technical reversal pattern detection")
        return result
        
    def visualize_patterns(self, df: pd.DataFrame, column: str = 'Close',
                          start_idx: int = None, end_idx: int = None,
                          save_path: Optional[str] = None) -> None:
        """
        Visualize detected technical reversal patterns.
        
        Args:
            df: DataFrame with pattern detections
            column: Column to analyze
            start_idx: Start index for visualization
            end_idx: End index for visualization
            save_path: Path to save visualization
        """
        # Subset data if requested
        if start_idx is not None or end_idx is not None:
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(df)
                
            plot_df = df.iloc[start_idx:end_idx].copy()
        else:
            plot_df = df.copy()
        
        plt.figure(figsize=(14, 10))
        
        # Plot price data
        plt.plot(plot_df.index, plot_df[column], color='blue', linewidth=1.5, label=column)
        
        # Mark peaks and valleys
        if 'is_peak' in plot_df.columns and 'is_valley' in plot_df.columns:
            peak_indices = plot_df[plot_df['is_peak']].index
            valley_indices = plot_df[plot_df['is_valley']].index
            
            plt.scatter(peak_indices, plot_df.loc[peak_indices, column], 
                       color='red', marker='^', s=50, label='Peaks')
            plt.scatter(valley_indices, plot_df.loc[valley_indices, column], 
                       color='green', marker='v', s=50, label='Valleys')
        
        # Mark Head and Shoulders patterns
        if 'hs_top' in plot_df.columns and 'hs_bottom' in plot_df.columns:
            hs_top_indices = plot_df[plot_df['hs_top']].index
            hs_bottom_indices = plot_df[plot_df['hs_bottom']].index
            
            plt.scatter(hs_top_indices, plot_df.loc[hs_top_indices, column], 
                       color='purple', marker='*', s=200, label='H&S Top')
            plt.scatter(hs_bottom_indices, plot_df.loc[hs_bottom_indices, column], 
                       color='orange', marker='*', s=200, label='H&S Bottom')
        
        # Mark Double Top/Bottom patterns
        if 'double_top' in plot_df.columns and 'double_bottom' in plot_df.columns:
            double_top_indices = plot_df[plot_df['double_top']].index
            double_bottom_indices = plot_df[plot_df['double_bottom']].index
            
            plt.scatter(double_top_indices, plot_df.loc[double_top_indices, column], 
                       color='darkred', marker='o', s=100, label='Double Top')
            plt.scatter(double_bottom_indices, plot_df.loc[double_bottom_indices, column], 
                       color='darkgreen', marker='o', s=100, label='Double Bottom')
        
        # Plot Support and Resistance levels
        if 'support_level' in plot_df.columns and 'resistance_level' in plot_df.columns:
            # Get unique support levels
            support_levels = plot_df['support_level'].dropna().unique()
            for level in support_levels:
                plt.axhline(y=level, color='green', linestyle='--', alpha=0.5)
            
            # Get unique resistance levels
            resistance_levels = plot_df['resistance_level'].dropna().unique()
            for level in resistance_levels:
                plt.axhline(y=level, color='red', linestyle='--', alpha=0.5)
        
        # Plot Fibonacci levels
        fib_columns = [col for col in plot_df.columns if col.startswith('fib_')]
        if fib_columns:
            # Get the last row's Fibonacci levels
            last_fib = plot_df[fib_columns].iloc[-1]
            
            # Plot each level
            for col, value in last_fib.items():
                if not np.isnan(value):
                    level = float(col.split('_')[1])
                    plt.axhline(y=value, color='purple', linestyle='-.', alpha=0.3)
                    plt.text(plot_df.index[0], value, f"Fib {level}", 
                            backgroundcolor='white', alpha=0.7)
        
        plt.title('Technical Reversal Patterns')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Technical reversal pattern visualization saved to {save_path}")
        else:
            plt.show()


class BreakoutDetection:
    """
    Breakout pattern detection.
    
    Detects breakout patterns including:
    - Volume-Confirmed Breakouts
    - Volatility Expansion Breakouts
    - Channel Breakouts
    """
    
    def __init__(self):
        """Initialize breakout pattern detector."""
        # Dictionary to store pattern detection results
        self.patterns = {}
        
    def detect_channel_breakouts(self, df: pd.DataFrame, column: str = 'Close',
                               lookback: int = 20, touch_count: int = 2) -> pd.DataFrame:
        """
        Detect price breakouts from established channels.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            lookback: Number of periods to establish channel
            touch_count: Minimum touches to establish valid channel
            
        Returns:
            DataFrame with breakout detections
        """
        result = df.copy()
        
        # Initialize columns
        result['upper_channel'] = np.nan
        result['lower_channel'] = np.nan
        result['channel_width'] = np.nan
        result['channel_breakout_up'] = False
        result['channel_breakout_down'] = False
        result['breakout_strength'] = 0.0
        
        # Process each data point
        for i in range(lookback, len(result)):
            # Define channel calculation window
            window = result.iloc[i-lookback:i]
            
            # Calculate upper and lower bounds of the channel
            upper_bound = window[column].max()
            lower_bound = window[column].min()
            
            # Count touches of the upper and lower bounds
            upper_touches = sum(abs(window[column] - upper_bound) < (0.01 * upper_bound))
            lower_touches = sum(abs(window[column] - lower_bound) < (0.01 * lower_bound))
            
            # Only establish a channel if there are enough touches
            if upper_touches >= touch_count and lower_touches >= touch_count:
                # Store channel bounds
                result.loc[result.index[i], 'upper_channel'] = upper_bound
                result.loc[result.index[i], 'lower_channel'] = lower_bound
                result.loc[result.index[i], 'channel_width'] = upper_bound - lower_bound
                
                # Check for breakouts
                current_price = result[column].iloc[i]
                
                # Upward breakout
                if current_price > upper_bound:
                    result.loc[result.index[i], 'channel_breakout_up'] = True
                    
                    # Calculate breakout strength
                    breakout_size = (current_price - upper_bound) / (upper_bound - lower_bound)
                    result.loc[result.index[i], 'breakout_strength'] = min(1.0, breakout_size)
                
                # Downward breakout
                elif current_price < lower_bound:
                    result.loc[result.index[i], 'channel_breakout_down'] = True
                    
                    # Calculate breakout strength
                    breakout_size = (lower_bound - current_price) / (upper_bound - lower_bound)
                    result.loc[result.index[i], 'breakout_strength'] = min(1.0, breakout_size)
        
        logger.info("Detected channel breakouts")
        return result
        
    def detect_volatility_breakouts(self, df: pd.DataFrame, column: str = 'Close',
                                  lookback: int = 20, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect breakouts based on volatility expansion.
        
        Args:
            df: DataFrame with price data
            column: Column to analyze
            lookback: Number of periods to establish baseline volatility
            threshold: Volatility multiple to trigger breakout
            
        Returns:
            DataFrame with volatility breakout detections
        """
        result = df.copy()
        
        # Calculate daily returns
        result['returns'] = result[column].pct_change()
        
        # Initialize columns
        result['volatility'] = np.nan
        result['volatility_breakout_up'] = False
        result['volatility_breakout_down'] = False
        result['volatility_breakout_strength'] = 0.0
        
        # Process each data point
        for i in range(lookback, len(result)):
            # Define volatility calculation window
            window = result.iloc[i-lookback:i]
            
            # Calculate historical volatility (standard deviation of returns)
            historical_vol = window['returns'].std()
            result.loc[result.index[i], 'volatility'] = historical_vol
            
            # Get current return
            current_return = result['returns'].iloc[i]
            
            # Check for volatility breakout
            if abs(current_return) > (historical_vol * threshold):
                if current_return > 0:
                    result.loc[result.index[i], 'volatility_breakout_up'] = True
                else:
                    result.loc[result.index[i], 'volatility_breakout_down'] = True
                
                # Calculate breakout strength
                breakout_ratio = abs(current_return) / (historical_vol * threshold)
                result.loc[result.index[i], 'volatility_breakout_strength'] = min(1.0, breakout_ratio)
        
        logger.info("Detected volatility breakouts")
        return result
        
    def detect_volume_breakouts(self, df: pd.DataFrame, price_col: str = 'Close',
                              volume_col: str = 'Volume', lookback: int = 20,
                              price_threshold: float = 0.02, 
                              volume_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect breakouts confirmed by volume expansion.
        
        Args:
            df: DataFrame with price and volume data
            price_col: Column for price data
            volume_col: Column for volume data
            lookback: Number of periods to establish baseline
            price_threshold: Minimum price movement to qualify
            volume_threshold: Volume multiple to confirm breakout
            
        Returns:
            DataFrame with volume-confirmed breakout detections
        """
        # Check if volume data exists
        if volume_col not in df.columns:
            logger.warning(f"Volume column '{volume_col}' not found. Cannot detect volume breakouts.")
            return df
            
        result = df.copy()
        
        # Initialize columns
        result['avg_volume'] = np.nan
        result['volume_breakout_up'] = False
        result['volume_breakout_down'] = False
        result['volume_ratio'] = np.nan
        
        # Process each data point
        for i in range(lookback, len(result)):
            # Define base window
            window = result.iloc[i-lookback:i]
            
            # Calculate average volume
            avg_volume = window[volume_col].mean()
            result.loc[result.index[i], 'avg_volume'] = avg_volume
            
            # Calculate volume ratio
            current_volume = result[volume_col].iloc[i]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            result.loc[result.index[i], 'volume_ratio'] = volume_ratio
            
            # Calculate price change
            price_change = result[price_col].iloc[i] / result[price_col].iloc[i-1] - 1
            
            # Check for volume-confirmed breakout
            if volume_ratio >= volume_threshold:
                if price_change >= price_threshold:
                    result.loc[result.index[i], 'volume_breakout_up'] = True
                elif price_change <= -price_threshold:
                    result.loc[result.index[i], 'volume_breakout_down'] = True
        
        logger.info("Detected volume-confirmed breakouts")
        return result
        
    def detect_all_breakouts(self, df: pd.DataFrame, price_col: str = 'Close',
                           volume_col: str = 'Volume') -> pd.DataFrame:
        """
        Detect all breakout patterns in one pass.
        
        Args:
            df: DataFrame with price data
            price_col: Column for price data
            volume_col: Column for volume data
            
        Returns:
            DataFrame with all breakout detections
        """
        logger.info("Detecting all breakout patterns")
        
        # Apply all breakout detection methods
        result = self.detect_channel_breakouts(df, column=price_col)
        result = self.detect_volatility_breakouts(result, column=price_col)
        
        # Only detect volume breakouts if volume data exists
        if volume_col in df.columns:
            result = self.detect_volume_breakouts(result, price_col=price_col, volume_col=volume_col)
        
        # Add a summary column for any breakout
        breakout_columns = [
            'channel_breakout_up', 'channel_breakout_down',
            'volatility_breakout_up', 'volatility_breakout_down'
        ]
        
        if volume_col in df.columns:
            breakout_columns.extend(['volume_breakout_up', 'volume_breakout_down'])
            
        up_columns = [col for col in breakout_columns if 'up' in col]
        down_columns = [col for col in breakout_columns if 'down' in col]
        
        result['breakout_up'] = result[up_columns].any(axis=1)
        result['breakout_down'] = result[down_columns].any(axis=1)
        result['has_breakout'] = result[breakout_columns].any(axis=1)
        
        logger.info("Completed breakout pattern detection")
        return result
        
    def visualize_breakouts(self, df: pd.DataFrame, price_col: str = 'Close',
                          volume_col: str = None, start_idx: int = None, 
                          end_idx: int = None, save_path: Optional[str] = None) -> None:
        """
        Visualize detected breakout patterns.
        
        Args:
            df: DataFrame with breakout detections
            price_col: Column for price data
            volume_col: Column for volume data
            start_idx: Start index for visualization
            end_idx: End index for visualization
            save_path: Path to save visualization
        """
        # Subset data if requested
        if start_idx is not None or end_idx is not None:
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = len(df)
                
            plot_df = df.iloc[start_idx:end_idx].copy()
        else:
            plot_df = df.copy()
        
        # Create figure with subplots
        fig_height = 10 if volume_col and volume_col in plot_df.columns else 8
        fig, axs = plt.subplots(2 if volume_col and volume_col in plot_df.columns else 1, 
                              1, figsize=(14, fig_height), 
                              gridspec_kw={'height_ratios': [3, 1]} if volume_col and volume_col in plot_df.columns else None)
        
        if volume_col and volume_col in plot_df.columns:
            price_ax = axs[0]
            volume_ax = axs[1]
        else:
            price_ax = axs if isinstance(axs, plt.Axes) else axs[0]
        
        # Plot price data
        price_ax.plot(plot_df.index, plot_df[price_col], color='blue', linewidth=1.5, label=price_col)
        
        # Plot channel if available
        if 'upper_channel' in plot_df.columns and 'lower_channel' in plot_df.columns:
            price_ax.plot(plot_df.index, plot_df['upper_channel'], color='red', 
                        linestyle='--', alpha=0.5, label='Upper Channel')
            price_ax.plot(plot_df.index, plot_df['lower_channel'], color='green', 
                        linestyle='--', alpha=0.5, label='Lower Channel')
        
        # Mark channel breakouts
        if 'channel_breakout_up' in plot_df.columns and 'channel_breakout_down' in plot_df.columns:
            breakout_up_indices = plot_df[plot_df['channel_breakout_up']].index
            breakout_down_indices = plot_df[plot_df['channel_breakout_down']].index
            
            price_ax.scatter(breakout_up_indices, plot_df.loc[breakout_up_indices, price_col], 
                        color='darkgreen', marker='^', s=100, label='Channel Breakout Up')
            price_ax.scatter(breakout_down_indices, plot_df.loc[breakout_down_indices, price_col], 
                        color='darkred', marker='v', s=100, label='Channel Breakout Down')
        
        # Mark volatility breakouts
        if 'volatility_breakout_up' in plot_df.columns and 'volatility_breakout_down' in plot_df.columns:
            vol_up_indices = plot_df[plot_df['volatility_breakout_up']].index
            vol_down_indices = plot_df[plot_df['volatility_breakout_down']].index
            
            price_ax.scatter(vol_up_indices, plot_df.loc[vol_up_indices, price_col], 
                        color='lime', marker='^', s=80, label='Volatility Breakout Up')
            price_ax.scatter(vol_down_indices, plot_df.loc[vol_down_indices, price_col], 
                        color='tomato', marker='v', s=80, label='Volatility Breakout Down')
        
        price_ax.set_title('Price Breakouts')
        price_ax.set_ylabel('Price')
        price_ax.grid(True, alpha=0.3)
        price_ax.legend(loc='upper left')
        
        # Plot volume if available
        if volume_col and volume_col in plot_df.columns:
            volume_ax.bar(plot_df.index, plot_df[volume_col], color='gray', alpha=0.5, label='Volume')
            
            # Add average volume line if available
            if 'avg_volume' in plot_df.columns:
                volume_ax.plot(plot_df.index, plot_df['avg_volume'], color='red', 
                              linestyle='-', alpha=0.7, label='Avg Volume')
            
            # Highlight volume breakout bars if available
            if 'volume_breakout_up' in plot_df.columns and 'volume_breakout_down' in plot_df.columns:
                vol_breakout_up = plot_df[plot_df['volume_breakout_up']]
                vol_breakout_down = plot_df[plot_df['volume_breakout_down']]
                
                volume_ax.bar(vol_breakout_up.index, vol_breakout_up[volume_col], 
                             color='green', alpha=0.7, label='Volume Breakout Up')
                volume_ax.bar(vol_breakout_down.index, vol_breakout_down[volume_col], 
                             color='red', alpha=0.7, label='Volume Breakout Down')
            
            volume_ax.set_title('Volume')
            volume_ax.set_ylabel('Volume')
            volume_ax.grid(True, alpha=0.3)
            volume_ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Breakout pattern visualization saved to {save_path}")
        else:
            plt.show()


class PatternRecognitionSystem:
    """
    Comprehensive pattern recognition system for forex data.
    
    Combines multiple pattern detection methods:
    - Candlestick patterns
    - Technical reversal patterns
    - Breakout detection
    """
    
    def __init__(self):
        """Initialize the pattern recognition system."""
        self.candlestick_patterns = CandlestickPatterns()
        self.reversal_patterns = TechnicalReversalPatterns()
        self.breakout_detector = BreakoutDetection()
        
        self.pattern_data = {}  # Store processed pattern data
        
    def detect_all_patterns(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply all pattern detection methods to multiple currency pairs.
        
        Args:
            data_dict: Dictionary of DataFrames for different currency pairs
            
        Returns:
            Dictionary of DataFrames with detected patterns
        """
        for pair_name, df in data_dict.items():
            logger.info(f"Detecting patterns for {pair_name}")
            
            # Detect candlestick patterns
            result_df = self.candlestick_patterns.detect_all_patterns(df)
            
            # Detect technical reversal patterns
            result_df = self.reversal_patterns.detect_all_patterns(result_df)
            
            # Detect breakouts
            result_df = self.breakout_detector.detect_all_breakouts(result_df)
            
            # Store the result
            self.pattern_data[pair_name] = result_df
            logger.info(f"Completed pattern detection for {pair_name}")
            
        return self.pattern_data
    
    def save_pattern_data(self, output_dir: str = 'data/patterns/') -> None:
        """
        Save pattern detection data to CSV files.
        
        Args:
            output_dir: Directory to save pattern data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for pair_name, df in self.pattern_data.items():
            output_path = os.path.join(output_dir, f"{pair_name}_patterns.csv")
            df.to_csv(output_path)
            logger.info(f"Saved pattern data for {pair_name} to {output_path}")
    
    def create_pattern_visualizations(self, pair_name: str, output_dir: str = 'results/figures/patterns/') -> None:
        """
        Create visualizations for detected patterns.
        
        Args:
            pair_name: Currency pair to visualize
            output_dir: Directory to save visualizations
        """
        if pair_name not in self.pattern_data:
            logger.error(f"No pattern data found for {pair_name}")
            return
            
        df = self.pattern_data[pair_name]
        os.makedirs(output_dir, exist_ok=True)
        
        # Create candlestick pattern visualization
        self.candlestick_patterns.visualize_patterns(
            df, 
            save_path=os.path.join(output_dir, f"{pair_name}_candlestick_patterns.png")
        )
        
        # Create technical reversal pattern visualization
        self.reversal_patterns.visualize_patterns(
            df, 
            save_path=os.path.join(output_dir, f"{pair_name}_reversal_patterns.png")
        )
        
        # Create breakout pattern visualization
        self.breakout_detector.visualize_breakouts(
            df, 
            save_path=os.path.join(output_dir, f"{pair_name}_breakout_patterns.png")
        )
        
        logger.info(f"Created pattern visualizations for {pair_name}")