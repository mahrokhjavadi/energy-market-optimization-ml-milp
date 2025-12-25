import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Load and preprocess an electricity price time series.

    Expected CSV columns:
        - timestamp : datetime or string convertible to datetime
        - price_eur_mwh : numeric electricity price

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame : Preprocessed dataset sorted by timestamp.
        pd.Series    : Daily grouped electricity prices (list of prices per day).
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = {"timestamp", "price_eur_mwh"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Input CSV must contain columns: {required_columns}"
        )

    # Convert timestamp to datetime
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    # Convert price to numeric
    data["price_eur_mwh"] = pd.to_numeric(
        data["price_eur_mwh"], errors="coerce"
    )

    # Drop invalid rows
    data = data.dropna(subset=["timestamp", "price_eur_mwh"])

    # Sort by time
    data_sorted = data.sort_values(by="timestamp")

    # Extract date for daily grouping
    data_sorted["date"] = data_sorted["timestamp"].dt.date

    # Group prices by day
    daily_prices = (
        data_sorted
        .groupby(data_sorted["date"].astype(str))["price_eur_mwh"]
        .apply(list)
    )

    return data_sorted, daily_prices