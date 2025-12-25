import os
import numpy as np
import pandas as pd


def generate_synthetic_energy_prices(
    start_date="2025-01-01",
    days=182,  # ~6 months
    output_dir="data",
    seed=42
):
    """
    Generate synthetic electricity price data at:
    - 15-minute resolution
    - 60-minute resolution (hourly)

    Column format is compatible with Q1, Q2, Q3 code:
        - timestamp
        - price_eur_mwh
    """

    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================
    # 1) Generate 15-minute timestamps
    # ======================================================
    timestamps_15min = pd.date_range(
        start=start_date,
        periods=days * 96,  # 96 intervals per day
        freq="15min"
    )

    n = len(timestamps_15min)

    # ======================================================
    # 2) Price components
    # ======================================================
    base_price = 50
    daily_cycle = 10 * np.sin(2 * np.pi * timestamps_15min.hour / 24)
    weekly_cycle = 5 * np.sin(2 * np.pi * timestamps_15min.dayofweek / 7)
    noise = np.random.normal(0, 3, n)

    # Rare spikes
    spikes = np.zeros(n)
    spike_idx = np.random.choice(n, size=int(0.01 * n), replace=False)
    spikes[spike_idx] = np.random.uniform(20, 80, len(spike_idx))

    prices_15min = base_price + daily_cycle + weekly_cycle + noise + spikes
    prices_15min = np.clip(prices_15min, 0, None)

    # ======================================================
    # 3) 15-minute DataFrame
    # ======================================================
    df_15min = pd.DataFrame({
        "timestamp": timestamps_15min,
        "price_eur_mwh": prices_15min.round(2)
    })

    # ======================================================
    # 4) 60-minute aggregation
    # ======================================================
    df_60min = (
        df_15min
        .set_index("timestamp")
        .resample("1H")
        .mean()
        .reset_index()
    )

    # ======================================================
    # 5) Save datasets
    # ======================================================
    df_15min.to_csv(
        os.path.join(output_dir, "synthetic_prices_15min.csv"),
        index=False
    )

    df_60min.to_csv(
        os.path.join(output_dir, "synthetic_prices_60min.csv"),
        index=False
    )

    print("✅ Synthetic datasets generated successfully")
    print("• 15-min data  → synthetic_prices_15min.csv")
    print("• 60-min data  → synthetic_prices_60min.csv")
    print(f"• Coverage    → {days} days (~6 months)")


if __name__ == "__main__":
    generate_synthetic_energy_prices()