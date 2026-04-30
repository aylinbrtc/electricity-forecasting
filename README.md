# Electricity Price Forecasting

Hourly and daily electricity price forecasting using classical ML ensemble and deep learning approaches, trained on energy production/consumption data combined with weather features.

## Problem

Two forecasting targets:
1. **Hourly** — predict the next hour's spot electricity price
2. **Daily** — predict the average price over the next 24 hours

## Data

- `energy_dataset.csv` — hourly electricity price, production, and consumption
- `weather_features.csv` — co-located weather readings

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
