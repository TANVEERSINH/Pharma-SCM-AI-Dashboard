"""
=============================================================
 Pharma-SCM-AI-Dashboard | Phase 2 — Exploratory Data Analysis
=============================================================
 Author      : Tanveersinh
 Program     : Masters in Digitalization & Transformation
 Description : Explores and visualizes all 4 synthetic
               datasets. Produces charts saved to
               06_results/charts/ folder.
 Run AFTER   : generate_data.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"]      = 120
plt.rcParams["savefig.dpi"]     = 150
plt.rcParams["figure.facecolor"] = "white"

# ── Paths ────────────────────────────────────────────────────
DATA_DIR   = "02_data/synthetic"
OUTPUT_DIR = "06_results/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Pharma SCM — Exploratory Data Analysis")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════

demand_df    = pd.read_csv(f"{DATA_DIR}/demand_data.csv",    parse_dates=["date"])
inventory_df = pd.read_csv(f"{DATA_DIR}/inventory_data.csv")
supplier_df  = pd.read_csv(f"{DATA_DIR}/supplier_data.csv")
kpi_df       = pd.read_csv(f"{DATA_DIR}/kpi_data.csv")

print(f"\n✅ Datasets loaded:")
print(f"   demand_data    : {demand_df.shape[0]:,} rows × {demand_df.shape[1]} cols")
print(f"   inventory_data : {inventory_df.shape[0]:,} rows × {inventory_df.shape[1]} cols")
print(f"   supplier_data  : {supplier_df.shape[0]:,} rows × {supplier_df.shape[1]} cols")
print(f"   kpi_data       : {kpi_df.shape[0]:,} rows × {kpi_df.shape[1]} cols")


# ════════════════════════════════════════════════════════════
# CHART 1 — Total Daily Demand Over 3 Years (all products)
# ════════════════════════════════════════════════════════════

print("\n[1/7] Plotting total daily demand over time...")

daily_total = demand_df.groupby("date")["demand_units"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_total["date"], daily_total["demand_units"],
        color="#185FA5", linewidth=0.8, alpha=0.7)

# 30-day rolling average
rolling = daily_total["demand_units"].rolling(30).mean()
ax.plot(daily_total["date"], rolling,
        color="#D85A30", linewidth=2.5, label="30-day rolling average")

# Shade disruption periods
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-03-31"),
           alpha=0.12, color="red",    label="COVID wave (demand spike)")
ax.axvspan(pd.Timestamp("2023-08-01"), pd.Timestamp("2023-09-15"),
           alpha=0.12, color="orange", label="Supply shortage")
ax.axvspan(pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31"),
           alpha=0.12, color="blue",   label="Flu outbreak")

ax.set_title("Total Daily Pharmaceutical Demand — 2022 to 2024", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Total Units Demanded")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_total_daily_demand.png")
plt.close()
print("   ✅ Saved: 01_total_daily_demand.png")


# ════════════════════════════════════════════════════════════
# CHART 2 — Demand per Product (monthly aggregated)
# ════════════════════════════════════════════════════════════

print("[2/7] Plotting demand per product...")

demand_df["month"] = demand_df["date"].dt.to_period("M").astype(str)
monthly_product = demand_df.groupby(["month", "product"])["demand_units"].sum().reset_index()
monthly_product["month_dt"] = pd.to_datetime(monthly_product["month"])

fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()
products = list(demand_df["product"].unique())
colors   = sns.color_palette("tab10", len(products))

for i, (product, color) in enumerate(zip(products, colors)):
    subset = monthly_product[monthly_product["product"] == product]
    axes[i].plot(subset["month_dt"], subset["demand_units"],
                 color=color, linewidth=2)
    axes[i].fill_between(subset["month_dt"], subset["demand_units"],
                          alpha=0.15, color=color)
    axes[i].set_title(product.replace("_", " "), fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[i].tick_params(axis="x", rotation=45, labelsize=8)

plt.suptitle("Monthly Demand by Product — 2022 to 2024",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_demand_per_product.png", bbox_inches="tight")
plt.close()
print("   ✅ Saved: 02_demand_per_product.png")


# ════════════════════════════════════════════════════════════
# CHART 3 — Average Monthly Demand Heatmap (seasonality)
# ════════════════════════════════════════════════════════════

print("[3/7] Plotting seasonality heatmap...")

demand_df["month_num"] = demand_df["date"].dt.month
demand_df["year"]      = demand_df["date"].dt.year

month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

heatmap_data = demand_df.groupby(["product","month_num"])["demand_units"].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index="product", columns="month_num", values="demand_units")
heatmap_pivot.columns = [month_names[m] for m in heatmap_pivot.columns]

# Normalize each row for better comparison across products
heatmap_norm = heatmap_pivot.div(heatmap_pivot.mean(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_norm, annot=False, cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Relative Demand (1.0 = average)"})
ax.set_title("Seasonal Demand Patterns by Product\n(darker = higher relative demand)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("")
ax.set_yticklabels([y.get_text().replace("_", " ") for y in ax.get_yticklabels()], fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_seasonality_heatmap.png")
plt.close()
print("   ✅ Saved: 03_seasonality_heatmap.png")


# ════════════════════════════════════════════════════════════
# CHART 4 — Inventory Status Distribution
# ════════════════════════════════════════════════════════════

print("[4/7] Plotting inventory status...")

status_counts = inventory_df["status"].value_counts()
colors_inv    = {"OK": "#2ecc71", "REORDER": "#f39c12", "CRITICAL": "#e74c3c"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
pie_colors = [colors_inv[s] for s in status_counts.index]
axes[0].pie(status_counts.values, labels=status_counts.index,
            autopct="%1.1f%%", colors=pie_colors,
            startangle=90, textprops={"fontsize": 12})
axes[0].set_title("Overall Inventory Status Distribution", fontsize=13, fontweight="bold")

# Bar chart — status by product
status_by_product = inventory_df.groupby(["product","status"]).size().unstack(fill_value=0)
status_by_product = status_by_product.reindex(columns=["OK","REORDER","CRITICAL"], fill_value=0)
bar_colors = [colors_inv[c] for c in status_by_product.columns]
status_by_product.plot(kind="bar", ax=axes[1], color=bar_colors, width=0.7)
axes[1].set_title("Inventory Status by Product (monthly count)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("Number of Months")
axes[1].tick_params(axis="x", rotation=45, labelsize=8)
axes[1].legend(title="Status")
axes[1].set_xticklabels([x.get_text().replace("_","\n") for x in axes[1].get_xticklabels()])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_inventory_status.png")
plt.close()
print("   ✅ Saved: 04_inventory_status.png")


# ════════════════════════════════════════════════════════════
# CHART 5 — Supplier Risk Scores
# ════════════════════════════════════════════════════════════

print("[5/7] Plotting supplier risk scores...")

supplier_sorted = supplier_df.sort_values("risk_score", ascending=True)
bar_colors_sup  = [colors_inv.get(r, "#95a5a6") for r in supplier_sorted["risk_category"]]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(supplier_sorted["supplier_name"], supplier_sorted["risk_score"],
               color=bar_colors_sup, edgecolor="white", height=0.6)

# Add value labels
for bar, val in zip(bars, supplier_sorted["risk_score"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=10)

ax.axvline(x=25, color="orange", linestyle="--", linewidth=1.5, label="Medium risk threshold (25)")
ax.axvline(x=50, color="red",    linestyle="--", linewidth=1.5, label="High risk threshold (50)")
ax.set_title("Supplier Risk Scores — Composite AI Risk Index (0 = safest, 100 = highest risk)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Risk Score")
ax.set_xlim(0, 80)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_supplier_risk.png")
plt.close()
print("   ✅ Saved: 05_supplier_risk.png")


# ════════════════════════════════════════════════════════════
# CHART 6 — AI vs Traditional Forecast Error Comparison
# ════════════════════════════════════════════════════════════

print("[6/7] Plotting AI vs traditional forecast error...")

kpi_df["month_dt"] = pd.to_datetime(kpi_df["month"])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(kpi_df["month_dt"], kpi_df["traditional_forecast_error_pct"],
        color="#e74c3c", linewidth=2.5, marker="o", markersize=3,
        label="Traditional method (Moving Average)")
ax.plot(kpi_df["month_dt"], kpi_df["ai_forecast_error_pct"],
        color="#2ecc71", linewidth=2.5, marker="o", markersize=3,
        label="AI model (Prophet)")
ax.fill_between(kpi_df["month_dt"],
                kpi_df["ai_forecast_error_pct"],
                kpi_df["traditional_forecast_error_pct"],
                alpha=0.15, color="#185FA5", label="Improvement area")

ax.set_title("Forecast Error: AI vs Traditional Methods — Monthly MAPE (%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Mean Absolute Percentage Error (%)")
ax.legend(fontsize=10)
ax.set_ylim(0, 35)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_ai_vs_traditional_forecast.png")
plt.close()
print("   ✅ Saved: 06_ai_vs_traditional_forecast.png")


# ════════════════════════════════════════════════════════════
# CHART 7 — Monthly Waste Reduction (AI vs Traditional)
# ════════════════════════════════════════════════════════════

print("[7/7] Plotting waste reduction chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart — waste units
x      = range(len(kpi_df))
width  = 0.4
labels = kpi_df["month"].str[-5:]  # last 5 chars e.g. "01-01"

axes[0].bar([i - width/2 for i in x], kpi_df["waste_units_traditional"],
            width, label="Traditional", color="#e74c3c", alpha=0.8)
axes[0].bar([i + width/2 for i in x], kpi_df["waste_units_ai"],
            width, label="AI-optimized", color="#2ecc71", alpha=0.8)
axes[0].set_title("Monthly Waste Units: AI vs Traditional", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Wasted Units")
axes[0].set_xticks(list(x)[::6])
axes[0].set_xticklabels(list(labels)[::6], rotation=45, fontsize=9)
axes[0].legend()

# Line chart — cumulative cost savings
kpi_df["cumulative_saving"] = kpi_df["cost_saving_eur"].cumsum()
axes[1].fill_between(kpi_df["month_dt"], kpi_df["cumulative_saving"],
                     alpha=0.3, color="#185FA5")
axes[1].plot(kpi_df["month_dt"], kpi_df["cumulative_saving"],
             color="#185FA5", linewidth=2.5)
axes[1].set_title("Cumulative Cost Savings from AI Optimization (EUR)",
                  fontsize=12, fontweight="bold")
axes[1].set_ylabel("Cumulative Savings (EUR)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_waste_reduction_savings.png")
plt.close()
print("   ✅ Saved: 07_waste_reduction_savings.png")


# ════════════════════════════════════════════════════════════
# EDA SUMMARY STATISTICS — printed to console
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  EDA SUMMARY STATISTICS")
print("=" * 60)

print("\n📦 DEMAND DATA:")
print(f"   Total records          : {len(demand_df):,}")
print(f"   Date range             : {demand_df['date'].min().date()} → {demand_df['date'].max().date()}")
print(f"   Products               : {demand_df['product'].nunique()}")
print(f"   Total units demanded   : {demand_df['demand_units'].sum():,}")
print(f"   Avg daily demand/product: {demand_df['demand_units'].mean():.0f} units")
print(f"   Max single day demand  : {demand_df['demand_units'].max():,} units")

print("\n🏭 INVENTORY DATA:")
critical_pct = (inventory_df["status"] == "CRITICAL").mean() * 100
reorder_pct  = (inventory_df["status"] == "REORDER").mean()  * 100
print(f"   Total monthly records  : {len(inventory_df)}")
print(f"   Critical events        : {(inventory_df['status']=='CRITICAL').sum()} ({critical_pct:.1f}%)")
print(f"   Reorder triggered      : {(inventory_df['status']=='REORDER').sum()} ({reorder_pct:.1f}%)")

print("\n🚚 SUPPLIER DATA:")
print(f"   Total suppliers        : {len(supplier_df)}")
print(f"   High risk suppliers    : {(supplier_df['risk_category']=='HIGH').sum()}")
print(f"   Medium risk suppliers  : {(supplier_df['risk_category']=='MEDIUM').sum()}")
print(f"   Low risk suppliers     : {(supplier_df['risk_category']=='LOW').sum()}")
print(f"   Avg risk score         : {supplier_df['risk_score'].mean():.1f}")

print("\n📊 KPI DATA:")
print(f"   Avg AI forecast error      : {kpi_df['ai_forecast_error_pct'].mean():.1f}%")
print(f"   Avg traditional error      : {kpi_df['traditional_forecast_error_pct'].mean():.1f}%")
print(f"   Avg improvement            : {kpi_df['forecast_improvement_pct'].mean():.1f} percentage points")
print(f"   Total simulated savings    : €{kpi_df['cost_saving_eur'].sum():,.2f}")
print(f"   Avg waste reduction        : {kpi_df['waste_reduction_pct'].mean():.1f}%")

print("\n" + "=" * 60)
print("  ✅ EDA complete! 7 charts saved to 06_results/charts/")
print("  Next step → run the AI models in 03_demand_forecasting.py")
print("=" * 60)
