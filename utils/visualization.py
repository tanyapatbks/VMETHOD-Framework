"""
Visualization utilities for the VMETHOD framework.

This module provides common visualization functions used across different stages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Visualization')

def setup_plot_style():
    """Set up matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_time_series(df: pd.DataFrame, columns: List[str], title: str = 'Time Series Plot',
                    y_label: str = 'Value', date_format: str = '%Y-%m',
                    fig_size: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
    """
    Create a time series plot for selected columns.
    
    Args:
        df: DataFrame containing time series data
        columns: List of column names to plot
        title: Plot title
        y_label: Y-axis label
        date_format: Format for date ticks
        fig_size: Figure size as (width, height) tuple
        save_path: Path to save the figure, if None, display only
    """
    setup_plot_style()
    
    plt.figure(figsize=fig_size)
    
    for column in columns:
        if column in df.columns:
            plt.plot(df.index, df[column], label=column)
        else:
            logger.warning(f"Column '{column}' not found in DataFrame")
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series plot saved to {save_path}")
    else:
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str = 'Correlation Matrix',
                          fig_size: Tuple[int, int] = (10, 8), save_path: Optional[str] = None):
    """
    Create a correlation matrix heatmap for DataFrame columns.
    
    Args:
        df: DataFrame containing data to analyze
        title: Plot title
        fig_size: Figure size as (width, height) tuple
        save_path: Path to save the figure, if None, display only
    """
    setup_plot_style()
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap
    plt.figure(figsize=fig_size)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 8})
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    else:
        plt.show()

def plot_regime_colored_series(df: pd.DataFrame, price_col: str = 'Close', regime_col: str = 'Close_regime',
                             title: str = 'Market Regimes', y_label: str = 'Price',
                             fig_size: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
    """
    Create a time series plot with background colored by market regime.
    
    Args:
        df: DataFrame containing price and regime data
        price_col: Column name for price data
        regime_col: Column name for regime classification
        title: Plot title
        y_label: Y-axis label
        fig_size: Figure size as (width, height) tuple
        save_path: Path to save the figure, if None, display only
    """
    if price_col not in df.columns or regime_col not in df.columns:
        logger.error(f"Required columns not found in DataFrame")
        return
    
    setup_plot_style()
    
    plt.figure(figsize=fig_size)
    
    # Plot price data
    plt.plot(df.index, df[price_col], color='blue', linewidth=1.5, label=price_col)
    
    # Color background by regime
    y_min, y_max = df[price_col].min(), df[price_col].max()
    y_range = y_max - y_min
    y_min -= 0.05 * y_range  # Add some padding
    y_max += 0.05 * y_range
    
    # Define regime colors
    colors = {
        1: 'green',   # Uptrend
        0: 'gray',    # Sideways
        -1: 'red'     # Downtrend
    }
    
    # Find contiguous regions of the same regime
    regime_data = df[regime_col].fillna(0).astype(int)
    
    for regime in colors.keys():
        # Create a mask for this regime
        mask = regime_data == regime
        if not mask.any():
            continue
            
        # Find start and end points of contiguous regions
        mask_indices = np.where(mask)[0]
        region_starts = [mask_indices[0]]
        region_ends = []
        
        for i in range(1, len(mask_indices)):
            if mask_indices[i] > mask_indices[i-1] + 1:
                region_ends.append(mask_indices[i-1])
                region_starts.append(mask_indices[i])
        
        region_ends.append(mask_indices[-1])
        
        # Fill each region
        for start, end in zip(region_starts, region_ends):
            start_date = df.index[start]
            end_date = df.index[end]
            plt.axvspan(start_date, end_date, alpha=0.2, color=colors[regime])
    
    # Add legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Uptrend'),
        Patch(facecolor='gray', alpha=0.2, label='Sideways'),
        Patch(facecolor='red', alpha=0.2, label='Downtrend'),
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label=price_col)
    ]
    plt.legend(handles=legend_elements)
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Regime-colored plot saved to {save_path}")
    else:
        plt.show()

def plot_wavelet_decomposition(df: pd.DataFrame, column: str = 'Close',
                             title: str = 'Wavelet Decomposition',
                             fig_size: Tuple[int, int] = (12, 10), save_path: Optional[str] = None):
    """
    Create a plot showing original series and its wavelet components.
    
    Args:
        df: DataFrame containing original and decomposed series
        column: Base column name
        title: Plot title
        fig_size: Figure size as (width, height) tuple
        save_path: Path to save the figure, if None, display only
    """
    # Check if wavelet decomposition columns exist
    trend_col = f'{column}_trend'
    details_col = f'{column}_details'
    
    if column not in df.columns or trend_col not in df.columns:
        logger.error(f"Required wavelet decomposition columns not found in DataFrame")
        return
    
    setup_plot_style()
    
    # Count detail columns
    detail_cols = [c for c in df.columns if c.startswith(f'{column}_detail_')]
    n_details = len(detail_cols)
    
    # Create figure with appropriate number of subplots
    fig, axs = plt.subplots(2 + n_details, 1, figsize=fig_size, sharex=True)
    
    # Plot original series
    axs[0].plot(df.index, df[column], label='Original')
    axs[0].set_title(f'Original {column} Series')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot trend
    axs[1].plot(df.index, df[trend_col], label='Trend', color='red')
    axs[1].set_title(f'Trend Component (Low Frequency)')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot detail levels
    for i, detail_col in enumerate(sorted(detail_cols)):
        axs[2+i].plot(df.index, df[detail_col], label=detail_col, color=f'C{i+2}')
        axs[2+i].set_title(f'Detail Component {i+1} (Higher Frequency)')
        axs[2+i].set_ylabel('Value')
        axs[2+i].legend()
        axs[2+i].grid(True, alpha=0.3)
    
    # Format date ticks
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Wavelet decomposition plot saved to {save_path}")
    else:
        plt.show()

def plot_training_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str = 'Close',
                           title: str = 'Train-Test Split', y_label: str = 'Price',
                           fig_size: Tuple[int, int] = (12, 6), save_path: Optional[str] = None):
    """
    Visualize train-test split for time series data.
    
    Args:
        train_df: Training DataFrame 
        test_df: Testing DataFrame
        column: Column to plot
        title: Plot title
        y_label: Y-axis label
        fig_size: Figure size as (width, height) tuple
        save_path: Path to save the figure, if None, display only
    """
    if column not in train_df.columns or column not in test_df.columns:
        logger.error(f"Column '{column}' not found in DataFrames")
        return
    
    setup_plot_style()
    
    plt.figure(figsize=fig_size)
    
    # Plot training data
    plt.plot(train_df.index, train_df[column], label='Training Data', color='blue')
    
    # Plot testing data
    plt.plot(test_df.index, test_df[column], label='Testing Data', color='red')
    
    # Add background colors
    min_val = min(train_df[column].min(), test_df[column].min())
    max_val = max(train_df[column].max(), test_df[column].max())
    buffer = (max_val - min_val) * 0.05
    
    # Background for training period
    if not train_df.empty:
        plt.axvspan(train_df.index[0], train_df.index[-1], alpha=0.1, color='blue')
    
    # Background for testing period
    if not test_df.empty:
        plt.axvspan(test_df.index[0], test_df.index[-1], alpha=0.1, color='red')
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Train-test split visualization saved to {save_path}")
    else:
        plt.show()