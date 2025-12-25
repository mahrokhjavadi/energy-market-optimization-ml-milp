import pandas as pd


def load_and_clean_data(file_path):
    """
    Load and clean electricity price data for exploratory analysis (EDA).

    Expected CSV columns:
        - timestamp
        - price_eur_mwh

    Returns:
        pd.DataFrame indexed by timestamp
        with a single standardized column 'price'
    """

    # Load CSV
    data = pd.read_csv(file_path)

    # ✅ Validate REAL CSV columns
    required_columns = {"timestamp", "price_eur_mwh"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    # Convert timestamp
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    # Convert price
    data["price"] = pd.to_numeric(
        data["price_eur_mwh"], errors="coerce"
    )

    # Drop invalid rows
    data = data.dropna(subset=["timestamp", "price"])

    # Set timestamp as index
    data = data.set_index("timestamp")

    # ✅ Return ONLY standardized column
    return data[["price"]]