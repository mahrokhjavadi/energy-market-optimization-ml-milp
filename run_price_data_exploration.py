import os
import pandas as pd

from preprocessing import load_and_preprocess_data
from analysis import calculate_statistics, calculate_correlation
from visualization import plot_line_chart, plot_box_plot, plot_histogram


def main():
    # =========================
    # Configuration
    # =========================
    file_path = "data/synthetic_prices.csv"
    output_folder = "outputs/price_data_exploration"

    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # Step 1: Load synthetic data (15-min resolution)
    # =========================
    data, _ = load_and_preprocess_data(file_path)

    data = data.rename(columns={
        "timestamp": "timestamp",
        "price_eur_mwh": "price"
    })

    data = data.set_index("timestamp")

    # =========================
    # Step 2: Create 15-min and 60-min datasets
    # =========================
    data_15min = data.copy()
    data_60min = data.resample("1H").mean()

    price_15min = data_15min["price"]
    price_60min = data_60min["price"]

    # =========================
    # Step 3: Descriptive statistics
    # =========================
    stats_15min = calculate_statistics(price_15min)
    stats_60min = calculate_statistics(price_60min)

    statistics_df = pd.DataFrame({
        "Statistic": stats_15min.keys(),
        "High-resolution (15-min)": stats_15min.values(),
        "Hourly (60-min)": stats_60min.values(),
    })

    statistics_df.to_csv(
        os.path.join(output_folder, "descriptive_statistics.csv"),
        index=False
    )

    # =========================
    # Step 4: Correlation analysis
    # =========================
    aligned_15min = price_15min.resample("1H").mean()
    correlation = calculate_correlation(aligned_15min, price_60min)

    with open(os.path.join(output_folder, "correlation.txt"), "w") as f:
        f.write(
            f"Correlation between high-resolution and hourly prices: {correlation:.2f}"
        )

    # =========================
    # Step 5: Visualizations
    # =========================
    plot_line_chart(
        price_15min,
        price_60min,
        data_15min,
        data_60min,
        output_folder,
    )

    plot_box_plot(price_15min, price_60min, output_folder)
    plot_histogram(price_15min, price_60min, output_folder)

    # =========================
    # Debug output
    # =========================
    print(f"Correlation: {correlation:.2f}")
    print(f"Results saved in: {output_folder}")


if __name__ == "__main__":
    main()