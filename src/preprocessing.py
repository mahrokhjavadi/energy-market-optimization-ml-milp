import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Load and preprocess an electricity price time series
    for Q1 and Q2 (hourly resolution).

    Expected CSV columns:
        - timestamp
        - price_eur_mwh

    Returns:
        pd.DataFrame : Hourly price time series
        pd.Series    : Daily prices (exactly 24 values per day)
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = {"timestamp", "price_eur_mwh"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Input CSV must contain columns: {required_columns}"
        )

    # Convert timestamp and price
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["price_eur_mwh"] = pd.to_numeric(data["price_eur_mwh"], errors="coerce")

    # Drop invalid rows
    data = data.dropna(subset=["timestamp", "price_eur_mwh"])

    # Sort by time
    data = data.sort_values("timestamp")

    # Extract date
    data["date"] = data["timestamp"].dt.date

    # Group prices by day
    daily_prices = (
        data.groupby("date")["price_eur_mwh"]
        .apply(list)
    )

    # ✅ CRITICAL FIX: keep only full 24-hour days
    daily_prices = daily_prices[daily_prices.apply(len) == 24]

    return data, daily_prices