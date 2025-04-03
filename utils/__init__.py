"""
Utility modules for the VMETHOD framework.
"""

from .visualization import (
    setup_plot_style,
    plot_time_series,
    plot_correlation_matrix,
    plot_regime_colored_series,
    plot_wavelet_decomposition,
    plot_training_test_split
)

__all__ = [
    'setup_plot_style',
    'plot_time_series',
    'plot_correlation_matrix',
    'plot_regime_colored_series',
    'plot_wavelet_decomposition',
    'plot_training_test_split'
]