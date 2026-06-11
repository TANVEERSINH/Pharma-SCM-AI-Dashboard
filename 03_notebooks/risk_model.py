"""
=============================================================
 Pharma-SCM-AI-Dashboard | Phase 3 — Risk Scoring Model
=============================================================
 Author      : Tanveersinh
 Program     : Masters in Digitalization & Transformation
 Description : ML-based supplier risk scoring model.
               Uses Random Forest to classify suppliers
               into risk categories and identify key
               risk drivers.
 Run AFTER   : inventory_model.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ── Paths ────────────────────────────────────────────────────
SUPPLIER_PATH = "02_data/synthetic/supplier_data.csv"
MODEL_DIR     = "04_models"
CHART_DIR     = "06_results/charts"
RESULT_DIR    = "06_results"

print("=" * 60)
print("  Pharma SCM — Supplier Risk Scoring Model")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(SUPPLIER_PATH)
print(f"\n✅ Supplier data loaded: {len(df)} suppliers")
print(f"   Risk distribution:\n{df['risk_category'].value_counts().to_string()}")


# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════

# Encode region
region_map = {"Europe": 1, "North America": 2, "Asia": 3,
              "Middle East": 4, "Africa": 5}
df["region_code"] = df["region"].map(region_map).fillna(3)

# Binary encode single_source
df["single_source_int"] = df["single_source"].astype(int)

# Reliability score (inverse of delay)
df["reliability_score"] = df["on_time_delivery_pct"] / 100

# Feature columns for ML model
FEATURES = [
    "delivery_delay_avg_days",
    "on_time_delivery_pct",
    "single_source_int",
    "financial_stability",
    "regulatory_compliance",
    "geopolitical_risk",
    "years_partnership",
    "region_code",
    "reliability_score",
]

X = df[FEATURES].values
y = df["risk_category"].values


# ════════════════════════════════════════════════════════════
# TRAIN RANDOM FOREST RISK CLASSIFIER
# ════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")
print("  Training Random Forest Risk Classifier...")
print(f"{'─'*60}")

# Encode labels
le    = LabelEncoder()
y_enc = le.fit_transform(y)   # HIGH=0, LOW=1, MEDIUM=2

# Scale features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf_model = RandomForestClassifier(
    n_estimators  = 200,
    max_depth     = 4,
    random_state  = 42,
    class_weight  = "balanced",
)
rf_model.fit(X_scaled, y_enc)

# Cross-validation (leave-one-out style since small dataset)
from sklearn.model_selection import LeaveOneOut
loo    = LeaveOneOut()
scores = cross_val_score(rf_model, X_scaled, y_enc, cv=loo, scoring="accuracy")
print(f"\n  Leave-One-Out CV Accuracy: {scores.mean()*100:.1f}%")

# Predict on full dataset (for demonstration)
y_pred     = rf_model.predict(X_scaled)
y_pred_lab = le.inverse_transform(y_pred)

# Feature importance
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

print(f"\n  Feature Importances (what drives risk most):")
for _, row in feat_imp_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"    {row['feature']:<30} {bar} {row['importance']:.3f}")


# ════════════════════════════════════════════════════════════
# ENHANCED RISK SCORING — combine ML + rule-based
# ════════════════════════════════════════════════════════════

# Get ML probability for HIGH risk
proba          = rf_model.predict_proba(X_scaled)
high_idx       = list(le.classes_).index("HIGH")
df["ml_high_risk_prob"] = proba[:, high_idx]

# Final risk score combining original score + ML probability
df["final_risk_score"] = (
    df["risk_score"]          * 0.6 +
    df["ml_high_risk_prob"]   * 40    # scale 0-1 → 0-40
).round(1)

df["ml_predicted_category"] = y_pred_lab

# Save enhanced supplier data
df.to_csv(f"{RESULT_DIR}/supplier_risk_scored.csv", index=False)
print(f"\n  ✅ Enhanced risk scores saved: {RESULT_DIR}/supplier_risk_scored.csv")


# ════════════════════════════════════════════════════════════
# SAVE MODELS
# ════════════════════════════════════════════════════════════

joblib.dump(rf_model, f"{MODEL_DIR}/risk_rf_model.pkl")
joblib.dump(le,       f"{MODEL_DIR}/risk_label_encoder.pkl")
joblib.dump(scaler,   f"{MODEL_DIR}/risk_scaler.pkl")
feat_imp_df.to_csv(f"{RESULT_DIR}/feature_importance.csv", index=False)
print(f"  ✅ Risk model saved: {MODEL_DIR}/risk_rf_model.pkl")


# ════════════════════════════════════════════════════════════
# CHART 1 — Feature Importance
# ════════════════════════════════════════════════════════════

print("\n  Generating risk model charts...")

fig, ax = plt.subplots(figsize=(10, 6))
colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(feat_imp_df)))
bars    = ax.barh(feat_imp_df["feature"][::-1],
                  feat_imp_df["importance"][::-1],
                  color=colors[::-1], edgecolor="white")

for bar, val in zip(bars, feat_imp_df["importance"][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10)

ax.set_title("Risk Model — Feature Importance\n(what factors drive supplier risk most)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Feature Importance Score")
ax.set_xlim(0, feat_imp_df["importance"].max() + 0.05)
clean_labels = [l.replace("_", " ").title() for l in feat_imp_df["feature"][::-1]]
ax.set_yticklabels(clean_labels, fontsize=10)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/12_feature_importance.png", dpi=150)
plt.close()
print(f"  ✅ Saved: {CHART_DIR}/12_feature_importance.png")


# ════════════════════════════════════════════════════════════
# CHART 2 — Final Supplier Risk Dashboard
# ════════════════════════════════════════════════════════════

df_sorted    = df.sort_values("final_risk_score", ascending=True)
color_map    = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
bar_colors   = [color_map.get(c, "#95a5a6") for c in df_sorted["risk_category"]]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(df_sorted["supplier_name"], df_sorted["final_risk_score"],
               color=bar_colors, edgecolor="white", height=0.6)

for bar, val in zip(bars, df_sorted["final_risk_score"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=10, fontweight="bold")

ax.axvline(x=25, color="orange", linestyle="--", linewidth=1.5)
ax.axvline(x=50, color="red",    linestyle="--", linewidth=1.5)
ax.text(25.5, 0.3, "Medium risk", color="orange", fontsize=9)
ax.text(50.5, 0.3, "High risk",   color="red",    fontsize=9)

patches = [
    mpatches.Patch(color="#2ecc71", label="LOW risk"),
    mpatches.Patch(color="#f39c12", label="MEDIUM risk"),
    mpatches.Patch(color="#e74c3c", label="HIGH risk"),
]
ax.legend(handles=patches, fontsize=10)
ax.set_title("Supplier Risk Scores — AI + ML Combined Risk Index\n(0 = safest, 100 = highest risk)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Final Risk Score")
ax.set_xlim(0, 85)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/13_supplier_risk_final.png", dpi=150)
plt.close()
print(f"  ✅ Saved: {CHART_DIR}/13_supplier_risk_final.png")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  RISK MODEL SUMMARY")
print("=" * 60)
print(f"\n  Suppliers analyzed       : {len(df)}")
print(f"  🔴 HIGH risk suppliers   : {(df['risk_category']=='HIGH').sum()}")
print(f"  🟡 MEDIUM risk suppliers : {(df['risk_category']=='MEDIUM').sum()}")
print(f"  🟢 LOW risk suppliers    : {(df['risk_category']=='LOW').sum()}")
print(f"\n  Top risk driver          : {feat_imp_df.iloc[0]['feature']}")
print(f"  Model CV Accuracy        : {scores.mean()*100:.1f}%")
print(f"\n  ⚠️  High risk suppliers to monitor:")
high_risk = df[df["risk_category"] == "HIGH"].sort_values("final_risk_score", ascending=False)
for _, row in high_risk.iterrows():
    print(f"     → {row['supplier_name']} ({row['country']}) | Score: {row['final_risk_score']}")
print(f"\n  ✅ All Phase 3 models complete!")
print(f"  ✅ Models saved to {MODEL_DIR}/")
print(f"  ✅ Results saved to {RESULT_DIR}/")
print(f"\n  Next step → build the dashboard (Phase 4)!")
print("=" * 60)
