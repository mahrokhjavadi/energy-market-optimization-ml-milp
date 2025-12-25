import os
import numpy as np
import pandas as pd


def generate_synthetic_energy_prices(
    start_date="2025-01-01",
    days=365,
    freq="1h",
    output_path="data/synthetic_prices.csv",
    seed=42
):
    """
    Generate synthetic day-ahead electricity prices.

    The data is fully artificial and intended for demos and portfolios.
    """

    np.random.seed(seed)

    # Create timestamp range
    timestamps = pd.date_range(
        start=start_date,
        periods=days * int(24 * 60 / 15),
        freq=freq
    )

    n = len(timestamps)

    # Price components
    base_price = 50
    daily_cycle = 10 * np.sin(2 * np.pi * timestamps.hour / 24)
    weekly_cycle = 5 * np.sin(2 * np.pi * timestamps.dayofweek / 7)
    noise = np.random.normal(0, 3, n)

    # Rare price spikes
    spikes = np.zeros(n)
    spike_indices = np.random.choice(n, size=int(0.01 * n), replace=False)
    spikes[spike_indices] = np.random.uniform(20, 80, len(spike_indices))

    prices = base_price + daily_cycle + weekly_cycle + noise + spikes
    prices = np.clip(prices, 0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price_eur_mwh": prices.round(2)
    })

    # ✅ THIS IS THE CRITICAL FIX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Synthetic data written to {output_path}")


if __name__ == "__main__":
    generate_synthetic_energy_prices()