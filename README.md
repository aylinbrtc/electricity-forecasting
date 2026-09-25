# Electricity Price Forecasting

Hourly and daily electricity price forecasting using classical ML ensemble and deep learning approaches, trained on energy production/consumption data combined with weather features.

## Problem

Two forecasting targets:
1. **Hourly** — predict the next hour's spot electricity price
2. **Daily** — predict the average price over the next 24 hours

## Data

Hourly data for the **Spanish electricity market**, covering **2015-01-01 to 2018-12-31**. Source: [Energy Consumption, Generation, Prices and Weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) (Kaggle, N. Jhana). The CSVs are not tracked in this repo; download them and place them in `data/raw/`.

- `energy_dataset.csv` — hourly market price (`price_actual`), generation by source, and total actual load
- `weather_features.csv` — weather readings for five Spanish cities (only Madrid is used, as it correlated most with price)

## Methodologies

### Method 1 — Classical ML Ensemble

Lag features, rolling statistics, and time-based features fed into MLP, LightGBM, and Random Forest models optimized with `RandomizedSearchCV` and validated with `TimeSeriesSplit`. A weighted ensemble outperforms individual models on the hourly task.

Notebooks in `notebooks/first_method/`:

| Notebook | Description |
|----------|-------------|
| `0_data_exploration` | Cyclicality, correlation analysis |
| `1_feature_engineering(_daily)` | Lag, rolling average, and time features |
| `2_model_training(_daily)` | Model training and hyperparameter tuning |
| `3_model_testing(_daily)` | Evaluation and ensemble construction |

### Method 2 — Deep Learning (Conv1D + Stacked GRU)

Signal-processing features (Fourier and wavelet transforms) fed into a Conv1D layer for local pattern extraction followed by stacked GRU layers for hierarchical temporal modeling. Hyperparameter search via KerasTuner.

Notebooks in `notebooks/second_method/`:

| Notebook | Description |
|----------|-------------|
| `1_feature_engineering_master` | Centralized preprocessing, outputs `.npz` sequences |
| `2_model_training_hourly` | GRU model for hourly prediction |
| `2_model_training_daily` | Conv1D + Stacked GRU for daily prediction |
| `3_model_testing_hourly/daily` | MAE, RMSE, R² evaluation |

## Results

Test set, original price scale (EUR).

| Task | Model | MAE | RMSE | R² |
|------|-------|-----|------|----|
| Hourly | Ensemble (MLP + LightGBM + RF)* | **1.88** | **2.70** | **0.961** |
| Hourly | LightGBM | 1.91 | 2.74 | 0.960 |
| Hourly | GRU baseline | 2.60 | 3.21 | 0.839 |
| Daily | LightGBM | **3.09** | **3.76** | **0.459** |
| Daily | Conv1D + Stacked GRU | 3.69 | 4.42 | 0.199 |
| Daily | Random Forest | 4.17 | 4.94 | 0.068 |
| Daily | MLP | 4.26 | 5.19 | -0.031 |

\* Scored on the **validation** set, which was also used to fit the ensemble weights (`3_model_testing.ipynb` loads `y_val.pkl`). Treat it as an optimistic estimate until re-evaluated on the held-out test split. The two methods also use different chronological splits, so the hourly ML/DL comparison is indicative only.

Naive baselines (`scripts/export_powerbi.py`, test split of the hourly series): last-hour price MAE 1.97 / R² 0.867; same hour yesterday MAE 3.93 / R² 0.531; daily average, previous day MAE 2.68 / R² 0.525. So the hourly persistence baseline is already at MAE ≈ 2, and the daily LightGBM (MAE 3.09) does not beat the previous-day baseline (MAE 2.68), although splits differ slightly.

The feature-engineered ensemble beats the deep learning models on both horizons. The Conv1D layer improves on a plain GRU for the daily task (which had a negative R²), but it does not beat LightGBM.

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Pre-trained models and processed datasets are available via the Google Drive links in the notebook headers.

## Project Structure

```
├── data/
│   ├── raw/               # energy_dataset.csv, weather_features.csv
│   └── processed/         # merged dataset, .npz sequences, scalers
├── models/                # best trained models (.keras, .joblib)
├── notebooks/
│   ├── first_method/      # Method 1 notebooks and saved models
│   └── second_method/     # Method 2 notebooks
└── requirements.txt
```
