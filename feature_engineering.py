def create_lag_features(df, lag_hours):
    """
    Create lagged price features for the dataset.

    Args:
        df (pd.DataFrame): Input data with a 'price' column.
        lag_hours (int): Number of lagged hours to create.

    Returns:
        pd.DataFrame: Dataset with added lagged features.
    """
    for lag in range(1, lag_hours + 1):
        df[f"lag_{lag}_hour"] = df["price"].shift(lag)
    return df


def create_rolling_features(df, rolling_window):
    """
    Create rolling statistics for the dataset.

    Args:
        df (pd.DataFrame): Input data with a 'price' column.
        rolling_window (int): Window size for rolling calculations.

    Returns:
        pd.DataFrame: Dataset with added rolling statistics.
    """
    df[f"rolling_mean_{rolling_window}"] = (
        df["price"].rolling(rolling_window).mean()
    )
    df[f"rolling_std_{rolling_window}"] = (
        df["price"].rolling(rolling_window).std()
    )
    return df