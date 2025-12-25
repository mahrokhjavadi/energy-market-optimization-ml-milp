import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# ======================================================
# MILP – DAILY PROFITS
# ======================================================
def plot_daily_profits(results_df, output_folder):
    """
    Plot daily optimization profits.
    """
    dates = pd.to_datetime(results_df["date"])
    profits = results_df["profit"]

    plt.figure(figsize=(14, 7))
    plt.plot(dates, profits, marker="o")
    plt.title("Daily Optimization Profit")
    plt.xlabel("Date")
    plt.ylabel("Profit (EUR)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_folder, "daily_profits.png")
    plt.savefig(path)
    plt.show()


# ======================================================
# MILP – 1 MWh STRATEGY
# ======================================================
def plot_strategy_1mwh(day_index, results, daily_prices, output_folder):
    """
    Plot charging/discharging strategy for 1 MWh battery.
    """
    result = results[day_index]
    date = result["date"]
    prices = daily_prices[date]

    hours = range(24)
    charge = result["Charge Schedule"]
    discharge = result["Discharge Schedule"]
    soc = result["SOC Schedule"]

    plt.figure(figsize=(14, 7))
    plt.plot(hours, prices, label="Price")
    plt.step(hours, soc, label="SOC", where="mid")

    plt.scatter(
        [h for h in hours if charge[h] == 1],
        [prices[h] for h in hours if charge[h] == 1],
        color="green",
        label="Charge",
        s=100,
    )

    plt.scatter(
        [h for h in hours if discharge[h] == 1],
        [prices[h] for h in hours if discharge[h] == 1],
        color="red",
        label="Discharge",
        s=100,
    )

    plt.title(f"Optimal Battery Strategy (1 MWh) – {date}")
    plt.xlabel("Hour")
    plt.ylabel("Price / SOC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_folder, f"strategy_1mwh_{date}.png")
    plt.savefig(path)
    plt.show()


# ======================================================
# MILP – 2 MWh STRATEGY (BLOCKING)
# ======================================================
def plot_strategy_2mwh_blocking(day_index, results, daily_prices, output_folder):
    """
    Plot charging/discharging strategy for 2 MWh battery
    with explicit full/half charge and discharge markers.
    """
    result = results[day_index]
    date = result["date"]
    prices = daily_prices[date]

    hours = list(range(24))
    soc = result["SOC Schedule"]

    charge_full = result["Charge Full Schedule"]
    charge_half = result["Charge Half Schedule"]
    discharge_full = result["Discharge Full Schedule"]
    discharge_half = result["Discharge Half Schedule"]

    plt.figure(figsize=(14, 7))

    # Price & SOC
    plt.plot(hours, prices, label="Hourly Prices (EUR/MWh)", linewidth=2)
    plt.step(hours, soc, label="SOC (MWh)", where="mid", linewidth=2)

    # ---- FULL / HALF CHARGE ----
    plt.scatter(
        [h for h in hours if charge_full[h] == 1],
        [prices[h] for h in hours if charge_full[h] == 1],
        color="green",
        s=120,
        label="Full Charge",
        zorder=5,
    )

    plt.scatter(
        [h for h in hours if charge_half[h] == 1],
        [prices[h] for h in hours if charge_half[h] == 1],
        color="lightgreen",
        s=120,
        label="Half Charge",
        zorder=5,
    )

    # ---- FULL / HALF DISCHARGE ----
    plt.scatter(
        [h for h in hours if discharge_full[h] == 1],
        [prices[h] for h in hours if discharge_full[h] == 1],
        color="red",
        s=120,
        label="Full Discharge",
        zorder=5,
    )

    plt.scatter(
        [h for h in hours if discharge_half[h] == 1],
        [prices[h] for h in hours if discharge_half[h] == 1],
        color="pink",
        s=120,
        label="Half Discharge",
        zorder=5,
    )

    # Labels & layout
    plt.title(f"Optimal Charging/Discharging Strategy (2 MWh) – {date}", fontsize=16)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Price (EUR/MWh) / SOC (MWh)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    path = os.path.join(output_folder, f"strategy_2mwh_{date}.png")
    plt.savefig(path)
    plt.show()
# ======================================================
# ML – FORECAST VS ACTUAL
# ======================================================
def plot_actual_vs_predicted(test_data, y_true, y_pred, output_folder):
    """
    Plot actual vs predicted prices.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(14, 7))
    plt.plot(test_data["timestamp"], y_true, label="Actual")
    plt.plot(test_data["timestamp"], y_pred, label="Predicted", linestyle="--")
    plt.title(f"Actual vs Predicted Prices (RMSE = {rmse:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_folder, "actual_vs_predicted.png")
    plt.savefig(path)
    plt.show()


# ======================================================
# ML + MILP – FORECAST-BASED STRATEGY
# ======================================================
def plot_strategy_forecast(day_index, results, test_data, output_folder):
    """
    Plot battery strategy based on forecasted prices.
    """
    result = results[day_index]
    date = result["date"]

    daily = test_data[test_data["date"] == date]
    prices = daily["predicted_price"].values
    hours = range(24)

    charge = result["Charge Schedule"]
    discharge = result["Discharge Schedule"]

    plt.figure(figsize=(14, 7))
    plt.plot(hours, prices, label="Forecasted Price")

    plt.scatter(
        [h for h in hours if charge[h] == 1],
        [prices[h] for h in hours if charge[h] == 1],
        color="green",
        label="Charge",
        s=100,
    )

    plt.scatter(
        [h for h in hours if discharge[h] == 1],
        [prices[h] for h in hours if discharge[h] == 1],
        color="red",
        label="Discharge",
        s=100,
    )

    plt.title(f"Forecast-based Strategy – {date}")
    plt.xlabel("Hour")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(output_folder, f"forecast_strategy_{date}.png")
    plt.savefig(path)
    plt.show()


# ======================================================
# EDA – PRICE ANALYSIS
# ======================================================
def plot_line_chart(price_15min, price_60min, data_15min, data_60min, output_folder):
    plt.figure(figsize=(12, 6))
    plt.plot(data_15min.index, price_15min, label="15-min")
    plt.plot(data_60min.index, price_60min, label="Hourly")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "line_prices.png"))
    plt.show()


def plot_box_plot(price_15min, price_60min, output_folder):
    plt.figure(figsize=(8, 5))
    plt.boxplot([price_15min, price_60min], labels=["15-min", "Hourly"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "boxplot_prices.png"))
    plt.show()


def plot_histogram(price_15min, price_60min, output_folder):
    plt.figure(figsize=(10, 6))
    sns.histplot(price_15min, bins=50, kde=True, label="15-min", alpha=0.6)
    sns.histplot(price_60min, bins=50, kde=True, label="Hourly", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "histogram_prices.png"))
    plt.show()