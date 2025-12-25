import numpy as np


def calculate_statistics(data):
    """
    Compute descriptive statistics for price data.
    """
    return {
        "Mean": data.mean(),
        "Median": data.median(),
        "Std Dev": data.std(),
        "Min": data.min(),
        "Max": data.max(),
        "25th Percentile": np.percentile(data, 25),
        "50th Percentile": np.percentile(data, 50),
        "75th Percentile": np.percentile(data, 75),
        "Variance": data.var(),
    }


def calculate_correlation(series1, series2):
    """
    Calculate correlation between two price series.
    """
    return series1.corr(series2)