# AGENTS.md — Google Antigravity Project Configuration

## Project Overview

* **Project Name**: HHS Unaccompanied Alien Children (UAC) Predictive Operations & Capacity Intelligence Suite

* **Stack**: Python 3.10+, Streamlit, Pandas, NumPy, SciPy, Scikit-Learn, Statsmodels, Plotly

* **Entrypoint**: `app.py` (`streamlit run app.py`)

* **Primary Dataset**: `HHS_Unaccompanied_Alien_Children_Program.csv`

## Architecture & Module Map

1. **`load_and_prepare_data()` (`app.py`)**:

   * Strips trailing blank rows from `HHS_Unaccompanied_Alien_Children_Program.csv`.

   * Parses `Date` (`2023-01-12` to `2025-12-21`, `720` observed reporting days) and reindexes to a continuous `1,075`-day daily frequency (`freq='D'`).

   * Interpolates weekend/holiday gaps (`355` days) via time-based interpolation while preserving an `is_observed` mask.

2. **`engineer_features()` (`app.py`)**:

   * Generates `72` predictive features: autoregressive lags (`t-1`, `t-2`, `t-3`, `t-7`, `t-14`), rolling statistics (`7d` and `14d` mean/std), `net_flow` (`Transfers - Discharges`), Little's Law Implied Length of Stay (`Active Care / 7d Rolling Discharges`), and Fourier calendar terms.

3. **`run_walk_forward_backtest()` (`app.py`)**:

   * Evaluates 7 models (`Random Forest Regressor`, `Gradient Boosting Regressor`, `ARIMA / SARIMA`, `Exponential Smoothing`, `Naive Persistence`, `Moving Average (7d)`, and `Mass-Balance Reconciled Ensemble`) over a 180-day out-of-sample walk-forward window using regime-aware local scaling.

   * Outputs `MAE`, `RMSE`, `MAPE (%)`, `Forecast Accuracy (%)`, multi-horizon errors (`1d`, `7d`, `14d`, `30d`), `Forecast Stability Index`, and `Model Robustness`.

4. **`generate_multi_horizon_forecasts()` (`app.py`)**:

   * Enforces physical mass-balance conservation: `Care(t) = Care(t-1) + Reconciled_Transfers(t) - Forecast_Discharges(t)`.

   * Computes calibrated `80%` and `95%` Conformal Prediction Intervals.

   * Runs a `1,000`-path Monte Carlo simulation to compute `Capacity Breach Probability (%)` and `Surge Lead Time (Days)`.

## Commands for Antigravity Agents

* **Install dependencies**: `pip install -r requirements.txt`

* **Run CLI self-test & backtest verification**: `python3 app.py`

* **Launch Streamlit dashboard**: `streamlit run app.py`