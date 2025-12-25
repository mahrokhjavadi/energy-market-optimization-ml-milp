import numpy as np


def calculate_statistics(data):
    """
    Compute descriptive statistics for a price time series.

    Statistics include mean, median, standard deviation,
    min, max, percentiles, and variance.

    Args:
        data (array-like or pd.Series): Price data.

    Returns:
        dict: Dictionary of calculated statistics.
    """
    return {
        "Mean": data.mean(),
        "Median": data.median(),
        "Std Dev": data.std(),
        "Min": data.min(),
        "Max": data.max(),
        "25th Percentile": np.percentile(data, 25),
        "50th Percentile (Median)": np.percentile(data, 50),
        "75th Percentile": np.percentile(data, 75),
        "Variance": data.var(),
    }


def calculate_correlation(series1, series2):
    """
    Calculate correlation between two aligned price series.

    Args:
        series1 (array-like or pd.Series)
        series2 (array-like or pd.Series)

    Returns:
        float: Pearson correlation coefficient.
    """
    return series1.corr(series2)