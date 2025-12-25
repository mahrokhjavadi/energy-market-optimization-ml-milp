import os
import pandas as pd

from src.preprocessing import load_and_preprocess_data
from src.optimization import optimize_battery_milp_2mwh_blocking
from src.visualization import plot_daily_profits, plot_strategy_2mwh_blocking


def main():
    # =========================
    # Configuration
    # =========================
    file_path = "data/synthetic_prices_60min.csv"
    output_folder = "outputs/milp_2mwh_blocking"
    n_days = 180
    day_index_to_plot = 50

    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # Step 1: Load and preprocess data
    # =========================
    _, daily_prices = load_and_preprocess_data(file_path)

    # Select first n_days
    test_daily_prices = daily_prices.head(n_days)

    # =========================
    # Step 2: Run MILP optimization
    # =========================
    results = []

    for date, prices in test_daily_prices.items():
        # Expect hourly prices (24 values per day)
        if len(prices) != 24 or pd.isnull(prices).any():
            print(f"Skipping {date} due to invalid data.")
            continue

        result = optimize_battery_milp_2mwh_blocking(prices)

        results.append({
            "date": date,
            "profit": result["Profit"],
            **result
        })

    # =========================
    # Step 3: Save results
    # =========================
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_folder, "results.csv"),
        index=False
    )

    # =========================
    # Step 4: Visualizations
    # =========================
    plot_daily_profits(results_df, output_folder)

    if day_index_to_plot < len(results):
        plot_strategy_2mwh_blocking(
            day_index_to_plot,
            results,
            test_daily_prices,
            output_folder
        )
    else:
        print("Selected day index out of range.")


if __name__ == "__main__":
    main()