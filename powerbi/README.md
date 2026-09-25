# Power BI report

Regenerate the tables with `python scripts/export_powerbi.py` (needs `data/raw/*.csv`).

## Load
Power BI Desktop → Get Data → Text/CSV → load both files in this folder:
- `hourly_prices.csv`: hourly Spanish price, load, three naive baselines, and a `split` column (train / validation / test).
- `model_metrics.csv`: MAE, RMSE, R² per model (copied from the notebook outputs).

In Power Query set `timestamp` to Date/Time and the numeric columns to Decimal Number.

## DAX measures (table `hourly_prices`)
```
MAE Last Hour   = AVERAGEX(FILTER(hourly_prices, NOT ISBLANK(hourly_prices[naive_last_hour])), ABS(hourly_prices[actual_price] - hourly_prices[naive_last_hour]))
MAE Yesterday   = AVERAGEX(FILTER(hourly_prices, NOT ISBLANK(hourly_prices[naive_same_hour_yesterday])), ABS(hourly_prices[actual_price] - hourly_prices[naive_same_hour_yesterday]))
MAE Last Week   = AVERAGEX(FILTER(hourly_prices, NOT ISBLANK(hourly_prices[naive_same_hour_last_week])), ABS(hourly_prices[actual_price] - hourly_prices[naive_same_hour_last_week]))
Avg Price       = AVERAGE(hourly_prices[actual_price])
Avg Load        = AVERAGE(hourly_prices[total_load])
```

## Pages
1. **Market**: line chart of `actual_price` by `timestamp` (day level), cards for Avg Price / Avg Load, slicer on year, hour-of-day × weekday matrix (average price) to show the daily/weekly seasonality.
2. **Model comparison**: clustered bar of `MAE` by `model`, legend = `family`, slicer on `task`; table with MAE / RMSE / R² / `evaluated_on`.
3. **Baselines**: slicer `split = test`; cards for MAE Last Hour / Yesterday / Last Week, to show what the models must beat.

## Predictions
`hourly_prices.csv` also has `pred_lightgbm` and `pred_random_forest` (Method 1, hourly), filled only for 2017-11-02 → 2018-05-30. This is the **validation** window the ensemble was scored on, not the test set. Rows were re-aligned to timestamps by rebuilding the merge (verified: LightGBM MAE 1.94 here vs 1.91 in the notebook). The MLP and the ensemble are not exported because `y_scaler.pkl` is missing. Deep learning predictions are not available yet (needs the `.keras` models and `X_test`).

Extra DAX, e.g. `MAE LightGBM = AVERAGEX(FILTER(hourly_prices, NOT ISBLANK(hourly_prices[pred_lightgbm])), ABS(hourly_prices[actual_price] - hourly_prices[pred_lightgbm]))`, and a line chart of `actual_price` vs `pred_lightgbm` for a one-week slice make a good page 2.
