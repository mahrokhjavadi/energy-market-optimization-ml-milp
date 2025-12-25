import os
import pandas as pd

from src.preprocessing_eda import load_and_clean_data
from src.analysis import calculate_statistics, calculate_correlation
from src.visualization import plot_line_chart, plot_box_plot, plot_histogram


def main():
    # =========================
    # Configuration
    # =========================
    output_folder = "outputs/price_data_exploration"
    os.makedirs(output_folder, exist_ok=True)

    file_15min = "data/synthetic_prices_15min.csv"
    file_60min = "data/synthetic_prices_60min.csv"

    # =========================
    # Step 1: Load data
    # (timestamp already index, price already unified)
    # =========================
    data_15min = load_and_clean_data(file_15min)
    data_60min = load_and_clean_data(file_60min)

    price_15min = data_15min["price"]
    price_60min = data_60min["price"]

    # =========================
    # Step 2: Descriptive statistics
    # =========================
    stats_15 = calculate_statistics(price_15min)
    stats_60 = calculate_statistics(price_60min)

    stats_df = pd.DataFrame({
        "Statistic": stats_15.keys(),
        "15-minute Prices": stats_15.values(),
        "60-minute Prices": stats_60.values(),
    })

    stats_df.to_csv(
        os.path.join(output_folder, "descriptive_statistics.csv"),
        index=False
    )

    # =========================
    # Step 3: Correlation analysis
    # =========================
    aligned_15min = price_15min.resample("1H").mean()
    correlation = calculate_correlation(aligned_15min, price_60min)

    with open(os.path.join(output_folder, "correlation.txt"), "w") as f:
        f.write(
            f"Correlation between 15-min (hourly aggregated) and 60-min prices: "
            f"{correlation:.4f}"
        )

    # =========================
    # Step 4: Visualizations
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
    # Console output
    # =========================
    print(f"Correlation: {correlation:.4f}")
    print(f"EDA results saved in: {output_folder}")


if __name__ == "__main__":
    main()