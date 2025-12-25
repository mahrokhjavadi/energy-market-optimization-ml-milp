import os
import pandas as pd

from preprocessing_ml import load_and_preprocess_data
from feature_engineering import create_lag_features, create_rolling_features
from modeling import train_lightgbm_model
from optimization import optimize_battery_milp_1mwh
from visualization import (
    plot_actual_vs_predicted,
    plot_daily_profits,
    plot_strategy_forecast,
)


def main():
    # =========================
    # Configuration
    # =========================
    file_path = "data/synthetic_prices.csv"
    output_folder = "outputs/ml_forecast_optimization"
    train_ratio = 0.8  # 80% train, 20% test

    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # Step 1: Load & preprocess data (ML-specific)
    # =========================
    data = load_and_preprocess_data(file_path)

    # Rename price column to ML-friendly name
    data = data.rename(columns={"price_eur_mwh": "price"})

    # Ensure correct temporal ordering
    data = data.sort_values("timestamp")

    # =========================
    # Step 2: Feature engineering
    # =========================
    data = create_lag_features(data, lag_hours=24)
    data = create_rolling_features(data, rolling_window=24)
    data.dropna(inplace=True)

    # =========================
    # Step 3: Train-test split
    # =========================
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    X_train = train_data.drop(["date", "timestamp", "price"], axis=1)
    y_train = train_data["price"]

    X_test = test_data.drop(["date", "timestamp", "price"], axis=1)
    y_test = test_data["price"]

    # =========================
    # Step 4: Train ML model
    # =========================
    model = train_lightgbm_model(X_train, y_train, X_test, y_test)

    test_data = test_data.copy()
    test_data["predicted_price"] = model.predict(X_test)

    # =========================
    # Step 5: MILP optimization using forecasts (1 MWh)
    # =========================
    daily_results = []

    for date, group in test_data.groupby("date"):
        prices = group["predicted_price"].values

        # MILP expects exactly 24 hourly prices
        if len(prices) != 24:
            continue

        result = optimize_battery_milp_1mwh(prices)
        result["date"] = date
        daily_results.append(result)

    results_df = pd.DataFrame(daily_results)
    results_df.to_csv(
        os.path.join(output_folder, "results.csv"),
        index=False,
    )

    # =========================
    # Step 6: Visualizations
    # =========================
    plot_actual_vs_predicted(
        test_data,
        y_test,
        test_data["predicted_price"],
        output_folder,
    )

    plot_daily_profits(results_df, output_folder)

    plot_strategy_forecast(
        day_index=0,
        results=daily_results,
        test_data=test_data,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    main()