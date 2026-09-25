"""Build Power BI input tables from the raw Kaggle data and the notebook metrics."""
import pandas as pd

e = pd.read_csv("data/raw/energy_dataset.csv", usecols=["time", "price actual", "total load actual"])
e["time"] = pd.to_datetime(e["time"], utc=True)
s = e.set_index("time").asfreq("h").interpolate()
n = len(s)

df = pd.DataFrame({
    "timestamp": s.index.tz_convert(None),
    "actual_price": s["price actual"].values,
    "total_load": s["total load actual"].values,
    "naive_last_hour": s["price actual"].shift(1).values,
    "naive_same_hour_yesterday": s["price actual"].shift(24).values,
    "naive_same_hour_last_week": s["price actual"].shift(168).values,
})
df["split"] = "train"
df.loc[int(n * 0.70):, "split"] = "validation"
df.loc[int(n * 0.85):, "split"] = "test"
df.to_csv("powerbi/hourly_prices.csv", index=False)

# Metrics copied from the notebook outputs (see notebooks/*/3_model_testing*).
# NOTE: the Method 1 hourly ensemble was scored on the validation set, not the test set.
rows = [
    ("Hourly", "Ensemble (MLP+LGBM+RF)", "ML", "validation", 1.88, 2.70, 0.961),
    ("Hourly", "Conv1D/GRU", "DL", "test", 2.60, 3.21, 0.839),
    ("Daily", "LightGBM", "ML", "test", 3.09, 3.76, 0.459),
    ("Daily", "Random Forest", "ML", "test", 4.17, 4.94, 0.068),
    ("Daily", "MLP", "ML", "test", 4.26, 5.19, -0.031),
    ("Daily", "Conv1D + Stacked GRU", "DL", "test", 3.69, 4.42, 0.199),
]
pd.DataFrame(rows, columns=["task", "model", "family", "evaluated_on", "MAE", "RMSE", "R2"]).to_csv(
    "powerbi/model_metrics.csv", index=False)

# --- Method 1 hourly predictions (validation split) from models/ensemble_info.pkl ---
# The notebooks scored these against y_val.pkl, which is not in the repo. Rows are re-aligned by
# rebuilding the energy+Madrid-weather merge (Madrid has duplicated timestamps, hence 36,267 rows)
# and taking the 70-85% slice; the offset was verified by MAE (minimum at row 25386, 1.97 EUR).
# MLP predictions are stored scaled and y_scaler.pkl is missing, so the ensemble is not rebuilt.
import os
if os.path.exists("models/ensemble_info.pkl"):
    import joblib
    info = joblib.load("models/ensemble_info.pkl")
    w = pd.read_csv("data/raw/weather_features.csv", usecols=["dt_iso", "city_name"])
    w = w[w.city_name == "Madrid"].assign(time=lambda d: pd.to_datetime(d.dt_iso, utc=True))
    m = e[["time", "price actual"]].merge(w[["time"]], on="time", how="left").sort_values("time", kind="stable").reset_index(drop=True)
    start = int(len(m) * 0.70)
    n_pred = len(info["lgbm_preds"])
    target_time = (m["time"].iloc[start:start + n_pred] + pd.Timedelta(hours=1)).dt.tz_convert(None)
    pr = pd.DataFrame({"timestamp": target_time.values,
                       "pred_lightgbm": info["lgbm_preds"],
                       "pred_random_forest": info["rf_preds"]}).groupby("timestamp").mean().reset_index()
    out = df.merge(pr, on="timestamp", how="left")
    out.to_csv("powerbi/hourly_prices.csv", index=False)
    v = out.dropna(subset=["pred_lightgbm"])
    print(len(v), v.timestamp.min(), v.timestamp.max(), (v.actual_price - v.pred_lightgbm).abs().mean())
