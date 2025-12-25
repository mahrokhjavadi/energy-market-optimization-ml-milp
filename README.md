ENERGY MARKET BATTERY OPTIMIZATION WITH MILP & MACHINE LEARNING
==============================================================

This repository contains a standalone research and portfolio project focused on
optimizing battery operation strategies in electricity markets using:

- Mixed-Integer Linear Programming (MILP)
- Machine Learning–based price forecasting
- Exploratory Data Analysis (EDA)
- Fully synthetic electricity price data

All data used in this project is artificially generated for demonstration,
experimentation, and educational purposes only.


--------------------------------------------------------------
PROJECT OVERVIEW
--------------------------------------------------------------

The project models a grid-connected battery participating in an electricity market.
At each time step, the battery can:

- Charge
- Discharge
- Remain idle

The objective is to determine optimal operational strategies that maximize economic
value while respecting realistic technical constraints such as:

- Battery power and energy limits
- State-of-charge dynamics
- No simultaneous charging and discharging
- Daily operational consistency

The project is modular and designed for experimentation, analysis, and extension.


--------------------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------------------

energy-market-optimization-ml-milp/

data/
  synthetic_prices_15min.csv
  synthetic_prices_60min.csv

outputs/
  price_data_exploration/
  milp_1mw_1mwh/
  milp_1mw_2mwh/
  ml_forecast_optimization/

src/
  __init__.py
  preprocessing.py          (optimization preprocessing)
  preprocessing_eda.py      (EDA preprocessing)
  preprocessing_ml.py       (ML preprocessing)
  feature_engineering.py
  modeling.py
  optimization.py
  analysis.py
  visualization.py

generate_synthetic_data.py
run_milp_battery_1mw_1mwh.py
run_milp_battery_1mw_2mwh_blocking.py
run_ml_forecast_optimization.py
run_price_data_exploration.py
README.txt


--------------------------------------------------------------
DATA DESCRIPTION
--------------------------------------------------------------

All electricity price data in this repository is synthetic.

Resolutions:
- 15-minute resolution (base data)
- 60-minute resolution (hourly aggregated data)

Time span:
- Approximately 6 months

CSV column format (used consistently across all scripts):

timestamp, price_eur_mwh


--------------------------------------------------------------
WORKFLOWS
--------------------------------------------------------------

1) SYNTHETIC DATA GENERATION
----------------------------

Generates artificial electricity prices with:
- Daily and weekly seasonal patterns
- Random noise
- Rare price spikes
- Multiple time resolutions

Run:
python generate_synthetic_data.py

Outputs:
- data/synthetic_prices_15min.csv
- data/synthetic_prices_60min.csv


2) BATTERY OPTIMIZATION WITH MILP
--------------------------------

Optimizes battery operation assuming perfect price information.

Battery configurations:
- 1 MW / 1 MWh
- 1 MW / 2 MWh (with blocking constraints)

Run (1 MWh):
python run_milp_battery_1mw_1mwh.py

Run (2 MWh with blocking):
python run_milp_battery_1mw_2mwh_blocking.py

Outputs include:
- Daily profit results (CSV)
- Charging and discharging schedules
- State-of-charge trajectories
- Strategy visualizations


3) FORECAST-BASED OPTIMIZATION (ML + MILP)
------------------------------------------

Combines machine learning price forecasting with battery optimization.

Machine Learning:
- Gradient boosting model (LightGBM)
- Features:
  - Lagged prices
  - Rolling statistics (mean, volatility)
- Target: next-hour electricity price

Optimization:
- MILP uses forecasted prices instead of perfect information

Run:
python run_ml_forecast_optimization.py

Outputs:
- Actual vs. predicted price plots
- Forecast-driven battery strategies
- Daily profit results


4) EXPLORATORY DATA ANALYSIS (EDA)
---------------------------------

Analyzes and compares electricity price behavior at different resolutions.

Analyses:
- Descriptive statistics
- Distribution comparison
- Correlation analysis

Visualizations:
- Time-series plots
- Box plots
- Histograms

Run:
python run_price_data_exploration.py

Outputs saved in:
outputs/price_data_exploration/


--------------------------------------------------------------
KEY INSIGHTS
--------------------------------------------------------------

- Higher-frequency prices show increased volatility
- Temporal aggregation smooths extreme price spikes
- Forecast uncertainty reduces achievable profits compared to perfect information
- MILP provides interpretable and reproducible decision logic


--------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------

Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- pulp
- lightgbm
- scikit-learn

Install:
pip install numpy pandas matplotlib seaborn pulp lightgbm scikit-learn

macOS (Apple Silicon):
brew install libomp


--------------------------------------------------------------
DESIGN PRINCIPLES
--------------------------------------------------------------

- Modular and reusable architecture
- Clear separation of preprocessing, modeling, optimization, and visualization
- Fully synthetic data for safe public sharing
- Reproducible and transparent workflows


--------------------------------------------------------------
POSSIBLE EXTENSIONS
--------------------------------------------------------------

- Risk-aware or multi-objective optimization
- Weather and demand feature integration
- Advanced forecasting models
- Multi-battery or portfolio-level optimization
- Stochastic optimization under uncertainty


--------------------------------------------------------------
AUTHOR
--------------------------------------------------------------

Mahrokh Javadi
PhD | Optimization & Data Science
Germany


--------------------------------------------------------------
NOTE
--------------------------------------------------------------

This project is intended for research, learning, and portfolio demonstration purposes.
