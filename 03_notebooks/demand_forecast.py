"""
=============================================================
 Pharma-SCM-AI-Dashboard | Phase 3 — Demand Forecasting Model
=============================================================
 Author      : Tanveersinh
 Program     : Masters in Digitalization & Transformation
 Description : Trains SARIMA + LinearRegression forecasting
               models for each pharmaceutical product.
               Compares AI vs traditional (moving average).
               No Prophet — works on all Python versions.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import joblib
import os
import json

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ────────────────────────────────────────────────────
DATA_PATH  = "02_data/synthetic/demand_data.csv"
MODEL_DIR  = "04_models"
CHART_DIR  = "06_results/charts"
RESULT_DIR = "06_results"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(CHART_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("  Pharma SCM — Demand Forecasting Model (ML Regression)")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
products = df["product"].unique()

print(f"\n✅ Data loaded: {len(df):,} rows | {len(products)} products")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Creates time-based features for ML model
# ════════════════════════════════════════════════════════════

def create_features(df_product):
    """
    Creates lag features and time features for ML forecasting.
    These replace the need for complex time-series models.
    """
    d = df_product.copy().sort_values("date").reset_index(drop=True)
    d["day_of_year"] = d["date"].dt.dayofyear
    d["month"]       = d["date"].dt.month
    d["quarter"]     = d["date"].dt.quarter
    d["year"]        = d["date"].dt.year
    d["day_of_week"] = d["date"].dt.dayofweek
    d["week"]        = d["date"].dt.isocalendar().week.astype(int)

    # Lag features — demand from past days
    for lag in [7, 14, 21, 30, 60, 90]:
        d[f"lag_{lag}"] = d["demand_units"].shift(lag)

    # Rolling mean features
    for window in [7, 14, 30]:
        d[f"rolling_mean_{window}"] = (
            d["demand_units"].shift(1).rolling(window).mean()
        )

    # Trend feature
    d["trend"] = np.arange(len(d))

    # Drop rows with NaN from lag features
    d = d.dropna().reset_index(drop=True)
    return d


def moving_average_forecast(train_y, periods, window=30):
    """Traditional baseline — simple moving average"""
    return np.full(periods, train_y.tail(window).mean())


def calculate_metrics(actual, predicted, model_name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    return {
        "model":    model_name,
        "MAE":      round(mae,  2),
        "RMSE":     round(rmse, 2),
        "MAPE_pct": round(mape, 2),
    }


# ════════════════════════════════════════════════════════════
# FEATURE COLUMNS
# ════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "day_of_year", "month", "quarter", "year",
    "day_of_week", "week", "trend",
    "lag_7", "lag_14", "lag_21", "lag_30", "lag_60", "lag_90",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
]


# ════════════════════════════════════════════════════════════
# TRAIN MODEL FOR EACH PRODUCT
# ════════════════════════════════════════════════════════════

all_results   = []
all_forecasts = {}

print(f"\n{'─'*60}")
print(f"  Training ML models for {len(products)} products...")
print(f"{'─'*60}")

for i, product in enumerate(products, 1):
    print(f"\n[{i}/{len(products)}] {product}")

    # Get product data and create features
    subset   = df[df["product"] == product][["date","demand_units"]].copy()
    featured = create_features(subset)

    # Train/test split — last 90 days as test
    split_idx = len(featured) - 90
    train     = featured.iloc[:split_idx]
    test      = featured.iloc[split_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train["demand_units"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["demand_units"].values

    # Scale features
    scaler   = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train ML model
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Predict
    y_pred = model.predict(X_test_sc)
    y_pred = np.maximum(y_pred, 0)   # no negative demand

    # Baseline — moving average
    baseline = moving_average_forecast(train["demand_units"], len(test))

    # Evaluate
    ml_metrics   = calculate_metrics(y_test, y_pred,   "ML Regression (AI)")
    base_metrics = calculate_metrics(y_test, baseline, "Moving Average (Traditional)")
    improvement  = base_metrics["MAPE_pct"] - ml_metrics["MAPE_pct"]

    print(f"   AI MAPE:   {ml_metrics['MAPE_pct']:.1f}%  |  "
          f"Traditional MAPE: {base_metrics['MAPE_pct']:.1f}%  |  "
          f"Improvement: {improvement:.1f}pp")

    # Save model
    joblib.dump(model,  f"{MODEL_DIR}/{product}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{product}_scaler.pkl")

    # Generate future 90-day forecast
    last_row  = featured.tail(1).copy()
    last_date = subset["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=90, freq="D"
    )

    # Simple future forecast using trained model
    future_preds = []
    last_demand  = subset["demand_units"].tail(90).values.tolist()

    for fd in future_dates:
        row = {
            "day_of_year": fd.dayofyear,
            "month":       fd.month,
            "quarter":     (fd.month - 1) // 3 + 1,
            "year":        fd.year,
            "day_of_week": fd.dayofweek,
            "week":        fd.isocalendar()[1],
            "trend":       len(featured) + len(future_preds),
            "lag_7":       last_demand[-7]  if len(last_demand) >= 7  else np.mean(last_demand),
            "lag_14":      last_demand[-14] if len(last_demand) >= 14 else np.mean(last_demand),
            "lag_21":      last_demand[-21] if len(last_demand) >= 21 else np.mean(last_demand),
            "lag_30":      last_demand[-30] if len(last_demand) >= 30 else np.mean(last_demand),
            "lag_60":      last_demand[-60] if len(last_demand) >= 60 else np.mean(last_demand),
            "lag_90":      last_demand[-90] if len(last_demand) >= 90 else np.mean(last_demand),
            "rolling_mean_7":  np.mean(last_demand[-7:]),
            "rolling_mean_14": np.mean(last_demand[-14:]),
            "rolling_mean_30": np.mean(last_demand[-30:]),
        }
        X_fut = scaler.transform([[row[c] for c in FEATURE_COLS]])
        pred  = max(0, model.predict(X_fut)[0])
        future_preds.append(pred)
        last_demand.append(pred)

    # Store forecast
    all_forecasts[product] = {
        "test_dates":    [str(d.date()) for d in test["date"]],
        "test_actual":   y_test.tolist(),
        "test_predicted": y_pred.tolist(),
        "future_dates":  [str(d.date()) for d in future_dates],
        "future_pred":   [round(p, 1) for p in future_preds],
    }

    all_results.append({
        "product":               product,
        "ai_mape_pct":           ml_metrics["MAPE_pct"],
        "ai_mae":                ml_metrics["MAE"],
        "ai_rmse":               ml_metrics["RMSE"],
        "traditional_mape_pct":  base_metrics["MAPE_pct"],
        "traditional_mae":       base_metrics["MAE"],
        "traditional_rmse":      base_metrics["RMSE"],
        "improvement_pp":        round(improvement, 2),
    })


# ════════════════════════════════════════════════════════════
# SAVE RESULTS
# ════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{RESULT_DIR}/model_evaluation.csv", index=False)

with open(f"{MODEL_DIR}/forecasts.json", "w") as f:
    json.dump(all_forecasts, f, indent=2)

print(f"\n{'─'*60}")
print(f"  ✅ Models saved to {MODEL_DIR}/")
print(f"  ✅ Evaluation saved to {RESULT_DIR}/model_evaluation.csv")


# ════════════════════════════════════════════════════════════
# CHART 1 — AI vs Traditional MAPE comparison
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))
x      = np.arange(len(results_df))
width  = 0.35
labels = [p.replace("_", "\n") for p in results_df["product"]]

bars1 = ax.bar(x - width/2, results_df["traditional_mape_pct"],
               width, label="Traditional (Moving Average)", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, results_df["ai_mape_pct"],
               width, label="AI — ML Regression",           color="#2ecc71", alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8, color="#e74c3c")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8, color="#27ae60")

ax.set_title("Forecast Accuracy: AI vs Traditional — MAPE % (lower is better)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("MAPE %")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_model_comparison.png", dpi=150)
plt.close()
print(f"  ✅ Saved: {CHART_DIR}/08_model_comparison.png")


# ════════════════════════════════════════════════════════════
# CHART 2 — Forecast vs Actual for Paracetamol
# ════════════════════════════════════════════════════════════

product  = "Paracetamol_500mg"
fc_data  = all_forecasts[product]
test_dates   = pd.to_datetime(fc_data["test_dates"])
future_dates = pd.to_datetime(fc_data["future_dates"])

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(test_dates, fc_data["test_actual"],
        color="#185FA5", linewidth=2, label="Actual demand")
ax.plot(test_dates, fc_data["test_predicted"],
        color="#D85A30", linewidth=2, linestyle="--", label="AI predicted")
ax.plot(future_dates, fc_data["future_pred"],
        color="#1D9E75", linewidth=2.5, label="Future forecast (90 days)")
ax.axvline(x=future_dates[0], color="black", linestyle=":", linewidth=1.5,
           label="Forecast start")

ax.set_title(f"AI Forecast vs Actual — Paracetamol 500mg",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Demand (Units)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_forecast_vs_actual.png", dpi=150)
plt.close()
print(f"  ✅ Saved: {CHART_DIR}/09_forecast_vs_actual.png")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 3 — MODEL EVALUATION SUMMARY")
print("=" * 60)
print(f"\n  {'Product':<25} {'AI MAPE':>9} {'Trad MAPE':>11} {'Improvement':>13}")
print(f"  {'─'*25} {'─'*9} {'─'*11} {'─'*13}")
for _, row in results_df.iterrows():
    print(f"  {row['product']:<25} "
          f"{row['ai_mape_pct']:>8.1f}% "
          f"{row['traditional_mape_pct']:>10.1f}% "
          f"{row['improvement_pp']:>+12.1f}pp")

avg_ai   = results_df["ai_mape_pct"].mean()
avg_trad = results_df["traditional_mape_pct"].mean()
avg_imp  = results_df["improvement_pp"].mean()

print(f"\n  {'AVERAGE':<25} {avg_ai:>8.1f}% {avg_trad:>10.1f}% {avg_imp:>+12.1f}pp")
print(f"\n  ✅ AI reduces forecast error by {avg_imp:.1f} percentage points")
print(f"  ✅ {len(results_df)} models saved to {MODEL_DIR}/")
print(f"  ✅ 2 charts saved to {CHART_DIR}/")
print(f"\n  Next step → run inventory_model.py")
print("=" * 60)
