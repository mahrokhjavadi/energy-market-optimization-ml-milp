import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Load electricity price data for ML-based forecasting.

    Expected CSV columns:
        - timestamp
        - price_eur_mwh

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Time-series dataset with columns
            ['date', 'timestamp', 'price']
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = {"timestamp", "price_eur_mwh"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    # Convert types
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["price"] = pd.to_numeric(
        data["price_eur_mwh"], errors="coerce"
    )

    # Drop invalid rows
    data = data.dropna(subset=["timestamp", "price"])

    # Add date column (used for grouping later)
    data["date"] = data["timestamp"].dt.date

    # Keep ML-relevant columns only
    return data[["date", "timestamp", "price"]]


def create_lag_features(df, lag_hours):
    """
    Add lagged price features.

    Args:
        df (pd.DataFrame): Input dataset with column 'price'.
        lag_hours (int): Number of lag hours.

    Returns:
        pd.DataFrame
    """
    for lag in range(1, lag_hours + 1):
        df[f"lag_{lag}_hour"] = df["price"].shift(lag)
    return df


def create_rolling_features(df, rolling_window):
    """
    Add rolling statistics of price.

    Args:
        df (pd.DataFrame): Input dataset with column 'price'.
        rolling_window (int): Rolling window size.

    Returns:
        pd.DataFrame
    """
    df[f"rolling_mean_{rolling_window}"] = (
        df["price"].rolling(rolling_window).mean()
    )
    df[f"rolling_std_{rolling_window}"] = (
        df["price"].rolling(rolling_window).std()
    )
    return df