# HHS Unaccompanied Alien Children (UAC) Predictive Operations & Capacity Intelligence Suite

An industry-grade, end-to-end **Time-Series Forecasting & Prescriptive Decision Intelligence** web application built with Python and Streamlit. Designed to transform the U.S. Department of Health and Human Services (HHS) UAC daily operations reporting from reactive descriptive tracking into proactive shelter capacity, medical staffing, and sponsor placement planning.

## 1. Quick Start

```
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place HHS_Unaccompanied_Alien_Children_Program.csv in the same folder as app.py (or upload via sidebar)
# 3. Launch the Streamlit web application
streamlit run app.py

```

You can also run the built-in CLI verification suite directly:

```
python3 app.py

```

## 2. Dataset & Time-Series Hygiene Summary

* **Source File**: `HHS_Unaccompanied_Alien_Children_Program.csv`

* **Raw Structure**: 1,170 rows × 6 columns (`450` trailing blank rows automatically stripped; `720` observed reporting dates from **January 12, 2023** to **December 21, 2025**).

* **Continuous Daily Reindexing**: Reindexed to a strict **1,075-day continuous daily grid** (`freq='D'`), handling weekend/holiday reporting gaps (`355` days) with time-based interpolation and an explicit `is_observed` binary mask.

* **Engineered Feature Space (72 Features)**:

  * **Autoregressive Lags**: `t-1`, `t-2`, `t-3`, `t-7`, and `t-14` across all 5 core series plus net flow.

  * **Rolling Moments**: 7-day and 14-day rolling means and standard deviations (shifted to prevent data leakage).

  * **Flow-Based Signals**: `Net Flow Pressure = Transfers − Discharges`, `Intake-to-Exit Ratio`, and `CBP Transfer Efficiency`.

  * **Queueing-Theory Length-of-Stay (`LoS`)**: Derived dynamically via Little's Law (`L = λW`, where `Implied LoS = Active HHS Care / 7d Rolling Discharge Rate`).

  * **Calendar & Cyclical Encodings**: Day-of-week, month, weekend indicators, and sine/cosine Fourier terms.

## 3. Walk-Forward Validation Leaderboard (180-Day Out-of-Sample Test)

All models are evaluated using strict chronological walk-forward validation over the final 180 days of the dataset with regime-aware local scaling to handle the structural shift between 2023–2024 (`6,000–11,516` care load) and 2025 (`1,972–2,528` care load).

### A. Active HHS Care Load (`Children in HHS Care`)

| Model | Family | MAE | RMSE | MAPE (%) | Forecast Accuracy (%) | Horizon 7d MAE | Horizon 14d MAE | Horizon 30d MAE | Forecast Stability Index | Model Robustness | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Random Forest Regressor** | Machine Learning | 4.61 | 5.94 | 0.21% | 99.79% | 5.85 | 6.44 | 7.34 | 98.7 | 99.8 | 
| **ARIMA / SARIMA** | Statistical | 4.82 | 6.21 | 0.22% | 99.78% | 6.59 | 7.43 | 8.71 | 95.5 | 99.8 | 
| **Mass-Balance Reconciled Ensemble** | Hybrid Reconciled | 5.09 | 6.53 | 0.23% | 99.77% | 6.46 | 7.11 | 8.11 | 99.4 | 99.8 | 
| **Gradient Boosting Regressor** | Machine Learning | 5.35 | 6.86 | 0.24% | 99.76% | 6.79 | 7.47 | 8.52 | 99.3 | 99.7 | 
| **Exponential Smoothing** | Statistical | 5.51 | 7.12 | 0.25% | 99.75% | 7.53 | 8.49 | 9.96 | 100.0 | 99.8 | 
| **Naive Persistence** | Baseline | 6.93 | 8.56 | 0.31% | 99.69% | 10.66 | 12.43 | 15.14 | 99.8 | 99.6 | 
| **Moving Average (7d)** | Baseline | 23.23 | 28.21 | 1.04% | 98.96% | 35.75 | 41.66 | 50.75 | 96.8 | 98.2 | 

### B. Daily Placement / Discharge Demand (`Children discharged from HHS Care`)

| Model | Family | MAE | RMSE | sMAPE (%) | Forecast Accuracy (%) | Horizon 7d MAE | Horizon 14d MAE | Horizon 30d MAE | Forecast Stability Index | Model Robustness | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Mass-Balance Reconciled Ensemble** | Hybrid Reconciled | 3.42 | 4.55 | 27.04% | 72.96% | 4.34 | 4.78 | 5.45 | 93.9 | 80.6 | 
| **Random Forest Regressor** | Machine Learning | 3.53 | 4.69 | 27.09% | 72.91% | 4.48 | 4.93 | 5.62 | 98.0 | 76.5 | 
| **ARIMA / SARIMA** | Statistical | 3.60 | 4.83 | 28.62% | 71.38% | 4.92 | 5.55 | 6.51 | 90.5 | 81.6 | 
| **Gradient Boosting Regressor** | Machine Learning | 3.75 | 4.99 | 29.61% | 70.39% | 4.76 | 5.24 | 5.97 | 99.4 | 76.1 | 
| **Naive Persistence** | Baseline | 3.96 | 5.25 | 31.43% | 68.57% | 6.09 | 7.10 | 8.65 | 99.9 | 74.9 | 
| **Moving Average (7d)** | Baseline | 4.05 | 5.32 | 31.51% | 68.49% | 6.23 | 7.26 | 8.85 | 85.0 | 80.2 | 
| **Exponential Smoothing** | Statistical | 4.10 | 5.39 | 32.31% | 67.69% | 5.61 | 6.32 | 7.41 | 91.1 | 78.2 | 

## 4. Commercial-Grade Features Included in `app.py`

1. **Mass-Balance Flow Reconciliation**: Enforces the physical stock-and-flow constraint `Care(t) = Care(t-1) + Reconciled_Transfers(t) − Forecast_Discharges(t)` so multi-series forecasts never contradict each other.

2. **Conformal Prediction Intervals (80% & 95%)**: Calibrates uncertainty fans directly from empirical walk-forward residuals scaled over the forecast horizon (`7` to `60` days).

3. **1,000-Path Monte Carlo Capacity Breach Engine**: Simulates stochastic inflow/outflow shocks to compute exact `Capacity Breach Probability (%)` and `Surge Lead Time (Days)`.

4. **Interactive Policy & Stress Scenario Simulator**: Compares Status Quo against Border Surge (`+35%` intake), Sponsor Vetting Bottleneck (`-25%` discharges), and Accelerated Placement (`+25%` discharges).

5. **Prescriptive Staffing & Budget Optimizer**: Translates forecasted upper-bound care loads into daily required shelter beds, caseworkers, medical personnel, and daily budget estimates, with one-click CSV export.