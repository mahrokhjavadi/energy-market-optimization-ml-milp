import pandas as pd


def load_and_clean_data(file_path):
    """
    Load and clean electricity price data for exploratory analysis (EDA).

    Expected CSV columns:
        - timestamp
        - price_eur_mwh

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset indexed by timestamp
                      with a single column 'price'.
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = {"timestamp", "price_eur_mwh"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    # Convert timestamp and price
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["price"] = pd.to_numeric(
        data["price_eur_mwh"], errors="coerce"
    )

    # Drop invalid rows
    data = data.dropna(subset=["timestamp", "price"])

    # Set timestamp as index (important for resampling & plotting)
    data = data.set_index("timestamp")

    return data[["price"]]