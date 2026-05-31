"""
=============================================================
 Pharma-SCM-AI-Dashboard | Phase 2 — Data Generation
=============================================================
 Author      : Tanveersinh
 Program     : Masters in Digitalization & Transformation
 Description : Generates synthetic pharmaceutical supply
               chain data including demand, inventory,
               supplier risk, and disruption events.
 Output      : 4 CSV files saved to 02_data/synthetic/
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)

# ── Output folder ────────────────────────────────────────────
OUTPUT_DIR = "02_data/synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Pharma SCM — Synthetic Data Generator")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# SECTION 1 — PRODUCT CATALOGUE
# 10 common pharmaceutical products with realistic profiles
# ════════════════════════════════════════════════════════════

PRODUCTS = {
    "Paracetamol_500mg": {
        "base_demand": 1200,   # units/day
        "unit_price":   0.05,  # EUR per unit
        "shelf_life":    730,  # days
        "category":   "OTC",
        "cold_chain": False,
        "seasonality": "winter_peak",   # high demand in winter (flu)
    },
    "Amoxicillin_250mg": {
        "base_demand":  400,
        "unit_price":   0.30,
        "shelf_life":   365,
        "category":   "Antibiotic",
        "cold_chain": False,
        "seasonality": "winter_peak",
    },
    "Metformin_500mg": {
        "base_demand":  800,
        "unit_price":   0.10,
        "shelf_life":  1095,
        "category":   "Diabetes",
        "cold_chain": False,
        "seasonality": "stable",        # chronic — steady demand
    },
    "Atorvastatin_20mg": {
        "base_demand":  600,
        "unit_price":   0.20,
        "shelf_life":  1095,
        "category":   "Cardiovascular",
        "cold_chain": False,
        "seasonality": "stable",
    },
    "Insulin_Glargine": {
        "base_demand":  150,
        "unit_price":  12.00,
        "shelf_life":   365,
        "category":   "Diabetes",
        "cold_chain": True,
        "seasonality": "stable",
    },
    "Ibuprofen_400mg": {
        "base_demand":  900,
        "unit_price":   0.08,
        "shelf_life":   730,
        "category":   "OTC",
        "cold_chain": False,
        "seasonality": "winter_peak",
    },
    "Omeprazole_20mg": {
        "base_demand":  500,
        "unit_price":   0.15,
        "shelf_life":   730,
        "category":   "Gastrointestinal",
        "cold_chain": False,
        "seasonality": "stable",
    },
    "Azithromycin_500mg": {
        "base_demand":  200,
        "unit_price":   0.80,
        "shelf_life":   730,
        "category":   "Antibiotic",
        "cold_chain": False,
        "seasonality": "winter_peak",
    },
    "Amlodipine_5mg": {
        "base_demand":  450,
        "unit_price":   0.12,
        "shelf_life":  1095,
        "category":   "Cardiovascular",
        "cold_chain": False,
        "seasonality": "stable",
    },
    "Cetirizine_10mg": {
        "base_demand":  350,
        "unit_price":   0.10,
        "shelf_life":   730,
        "category":   "Antihistamine",
        "cold_chain": False,
        "seasonality": "spring_peak",   # high in allergy season
    },
}


# ════════════════════════════════════════════════════════════
# SECTION 2 — DEMAND DATA GENERATION
# 3 years of daily demand: Jan 2022 – Dec 2024
# Includes: seasonality, trend, noise, COVID spike, disruptions
# ════════════════════════════════════════════════════════════

def get_seasonality_multiplier(date, pattern):
    """
    Returns a demand multiplier based on the season and pattern.
    winter_peak  → high Dec–Feb (flu season)
    spring_peak  → high Mar–May (allergy season)
    stable       → flat all year
    """
    month = date.month

    if pattern == "winter_peak":
        # Peak in Dec, Jan, Feb — drops in summer
        seasonal = {
            1: 1.40, 2: 1.35, 3: 1.10, 4: 0.90,
            5: 0.85, 6: 0.80, 7: 0.82, 8: 0.85,
            9: 0.95, 10: 1.05, 11: 1.25, 12: 1.45
        }
    elif pattern == "spring_peak":
        seasonal = {
            1: 0.85, 2: 0.90, 3: 1.20, 4: 1.45,
            5: 1.40, 6: 1.15, 7: 0.90, 8: 0.85,
            9: 0.90, 10: 1.00, 11: 0.95, 12: 0.85
        }
    else:  # stable
        seasonal = {m: 1.0 for m in range(1, 13)}

    return seasonal[month]


def get_trend_multiplier(date, start_date):
    """
    Slight upward trend of ~5% per year
    (population growth + chronic disease increase)
    """
    days_elapsed = (date - start_date).days
    annual_growth = 0.05
    return 1 + (annual_growth * days_elapsed / 365)


def get_disruption_multiplier(date):
    """
    Simulates real-world disruption events that spike or crash demand.
    COVID wave: Jan–Mar 2022 — demand for OTC drugs spikes 2.5x
    Supply shortage: Aug 2023 — demand drops 30% (stock unavailable)
    Flu outbreak: Nov–Dec 2024 — demand spike 1.8x
    """
    disruptions = {
        # (start, end, multiplier, label)
        "covid_wave":      (datetime(2022, 1, 1),  datetime(2022, 3, 31), 2.5),
        "supply_shortage": (datetime(2023, 8, 1),  datetime(2023, 9, 15), 0.7),
        "flu_outbreak":    (datetime(2024, 11, 1), datetime(2024, 12, 31), 1.8),
    }
    for label, (start, end, mult) in disruptions.items():
        if start <= date <= end:
            return mult, label
    return 1.0, "none"


def generate_demand_data():
    print("\n[1/4] Generating daily demand data (3 years × 10 products)...")

    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    records = []

    for product_name, profile in PRODUCTS.items():
        base   = profile["base_demand"]
        season = profile["seasonality"]

        for date in date_range:
            # Build demand from components
            trend_mult   = get_trend_multiplier(date, start_date)
            season_mult  = get_seasonality_multiplier(date, season)
            disrupt_mult, disrupt_label = get_disruption_multiplier(date)

            # Only OTC and Antibiotics are strongly affected by disruptions
            if profile["category"] not in ["OTC", "Antibiotic"]:
                disrupt_mult = 1 + (disrupt_mult - 1) * 0.3

            # Random daily noise ±8%
            noise = np.random.normal(1.0, 0.08)

            # Final demand (rounded to integer units)
            demand = int(base * trend_mult * season_mult * disrupt_mult * noise)
            demand = max(0, demand)  # no negative demand

            # Weekend effect — 20% lower on weekends (hospital ordering)
            if date.weekday() >= 5:
                demand = int(demand * 0.80)

            records.append({
                "date":             date.strftime("%Y-%m-%d"),
                "product":          product_name,
                "category":         profile["category"],
                "demand_units":     demand,
                "unit_price_eur":   profile["unit_price"],
                "demand_value_eur": round(demand * profile["unit_price"], 2),
                "cold_chain":       profile["cold_chain"],
                "disruption_event": disrupt_label,
                "seasonality_type": season,
            })

    df = pd.DataFrame(records)
    path = f"{OUTPUT_DIR}/demand_data.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ Saved: {path}")
    print(f"   Rows: {len(df):,} | Products: {df['product'].nunique()} | Date range: 2022–2024")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 3 — INVENTORY DATA GENERATION
# Stock levels, reorder points, safety stock per product
# ════════════════════════════════════════════════════════════

def generate_inventory_data():
    print("\n[2/4] Generating inventory data...")

    inventory_profiles = {
        "Paracetamol_500mg":  {"reorder_point": 15000, "safety_stock": 8000,  "max_stock": 60000, "lead_time_days": 7},
        "Amoxicillin_250mg":  {"reorder_point":  5000, "safety_stock": 2500,  "max_stock": 20000, "lead_time_days": 10},
        "Metformin_500mg":    {"reorder_point": 10000, "safety_stock": 5000,  "max_stock": 40000, "lead_time_days": 7},
        "Atorvastatin_20mg":  {"reorder_point":  7500, "safety_stock": 3500,  "max_stock": 30000, "lead_time_days": 10},
        "Insulin_Glargine":   {"reorder_point":  2000, "safety_stock": 1000,  "max_stock":  8000, "lead_time_days": 14},
        "Ibuprofen_400mg":    {"reorder_point": 12000, "safety_stock": 6000,  "max_stock": 45000, "lead_time_days": 7},
        "Omeprazole_20mg":    {"reorder_point":  6000, "safety_stock": 3000,  "max_stock": 25000, "lead_time_days": 7},
        "Azithromycin_500mg": {"reorder_point":  2500, "safety_stock": 1200,  "max_stock": 10000, "lead_time_days": 10},
        "Amlodipine_5mg":     {"reorder_point":  5500, "safety_stock": 2800,  "max_stock": 22000, "lead_time_days": 7},
        "Cetirizine_10mg":    {"reorder_point":  4500, "safety_stock": 2000,  "max_stock": 18000, "lead_time_days": 7},
    }

    # Monthly snapshots of stock levels
    months = pd.date_range(start="2022-01-01", end="2024-12-01", freq="MS")
    records = []

    for product, profile in inventory_profiles.items():
        base       = PRODUCTS[product]["base_demand"]
        stock      = profile["max_stock"] * 0.7  # start at 70% capacity

        for month in months:
            # Monthly consumption based on base demand × 30 days
            monthly_consumption = base * 30 * np.random.uniform(0.85, 1.15)

            # If stock drops below reorder point → replenish
            if stock < profile["reorder_point"]:
                replenishment = profile["max_stock"] - stock
                stock += replenishment
                reorder_triggered = True
            else:
                reorder_triggered = False

            stock -= monthly_consumption
            stock  = max(0, stock)

            # Determine status
            if stock < profile["safety_stock"]:
                status = "CRITICAL"
            elif stock < profile["reorder_point"]:
                status = "REORDER"
            else:
                status = "OK"

            days_until_stockout = int(stock / (base if base > 0 else 1))

            records.append({
                "month":              month.strftime("%Y-%m"),
                "product":            product,
                "current_stock":      int(stock),
                "safety_stock":       profile["safety_stock"],
                "reorder_point":      profile["reorder_point"],
                "max_stock":          profile["max_stock"],
                "lead_time_days":     profile["lead_time_days"],
                "days_until_stockout": days_until_stockout,
                "reorder_triggered":  reorder_triggered,
                "status":             status,
            })

    df = pd.DataFrame(records)
    path = f"{OUTPUT_DIR}/inventory_data.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ Saved: {path}")
    print(f"   Rows: {len(df):,} | Critical events: {(df['status']=='CRITICAL').sum()}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 4 — SUPPLIER DATA GENERATION
# 8 suppliers with risk scores across multiple dimensions
# ════════════════════════════════════════════════════════════

def generate_supplier_data():
    print("\n[3/4] Generating supplier risk data...")

    suppliers = [
        {
            "supplier_id":        "SUP001",
            "supplier_name":      "SunPharma India",
            "country":            "India",
            "region":             "Asia",
            "lat": 19.076,        "lon": 72.877,
            "products_supplied":  ["Paracetamol_500mg", "Metformin_500mg", "Ibuprofen_400mg"],
            "delivery_delay_avg_days": 3.2,
            "on_time_delivery_pct":    88,
            "single_source":           True,   # only supplier for some products
            "financial_stability":      7,      # score out of 10
            "regulatory_compliance":    8,
            "geopolitical_risk":        6,
            "years_partnership":        8,
        },
        {
            "supplier_id":        "SUP002",
            "supplier_name":      "Roche Basel",
            "country":            "Switzerland",
            "region":             "Europe",
            "lat": 47.559,        "lon": 7.588,
            "products_supplied":  ["Insulin_Glargine", "Atorvastatin_20mg"],
            "delivery_delay_avg_days": 1.1,
            "on_time_delivery_pct":    97,
            "single_source":           False,
            "financial_stability":      10,
            "regulatory_compliance":    10,
            "geopolitical_risk":        2,
            "years_partnership":        15,
        },
        {
            "supplier_id":        "SUP003",
            "supplier_name":      "Sinopharm China",
            "country":            "China",
            "region":             "Asia",
            "lat": 39.904,        "lon": 116.407,
            "products_supplied":  ["Amoxicillin_250mg", "Azithromycin_500mg", "Cetirizine_10mg"],
            "delivery_delay_avg_days": 5.8,
            "on_time_delivery_pct":    79,
            "single_source":           True,
            "financial_stability":      6,
            "regulatory_compliance":    7,
            "geopolitical_risk":        8,
            "years_partnership":        5,
        },
        {
            "supplier_id":        "SUP004",
            "supplier_name":      "Teva Pharmaceuticals",
            "country":            "Israel",
            "region":             "Middle East",
            "lat": 32.086,        "lon": 34.768,
            "products_supplied":  ["Omeprazole_20mg", "Amlodipine_5mg"],
            "delivery_delay_avg_days": 2.5,
            "on_time_delivery_pct":    91,
            "single_source":           False,
            "financial_stability":      8,
            "regulatory_compliance":    9,
            "geopolitical_risk":        7,
            "years_partnership":        10,
        },
        {
            "supplier_id":        "SUP005",
            "supplier_name":      "Cipla Mumbai",
            "country":            "India",
            "region":             "Asia",
            "lat": 18.975,        "lon": 72.826,
            "products_supplied":  ["Paracetamol_500mg", "Ibuprofen_400mg"],
            "delivery_delay_avg_days": 4.0,
            "on_time_delivery_pct":    84,
            "single_source":           False,
            "financial_stability":      7,
            "regulatory_compliance":    8,
            "geopolitical_risk":        6,
            "years_partnership":        6,
        },
        {
            "supplier_id":        "SUP006",
            "supplier_name":      "Novartis Germany",
            "country":            "Germany",
            "region":             "Europe",
            "lat": 48.135,        "lon": 11.582,
            "products_supplied":  ["Metformin_500mg", "Atorvastatin_20mg"],
            "delivery_delay_avg_days": 1.5,
            "on_time_delivery_pct":    95,
            "single_source":           False,
            "financial_stability":      10,
            "regulatory_compliance":    10,
            "geopolitical_risk":        1,
            "years_partnership":        12,
        },
        {
            "supplier_id":        "SUP007",
            "supplier_name":      "BioPharm Egypt",
            "country":            "Egypt",
            "region":             "Africa",
            "lat": 30.033,        "lon": 31.233,
            "products_supplied":  ["Cetirizine_10mg", "Omeprazole_20mg"],
            "delivery_delay_avg_days": 8.5,
            "on_time_delivery_pct":    68,
            "single_source":           True,
            "financial_stability":      4,
            "regulatory_compliance":    5,
            "geopolitical_risk":        9,
            "years_partnership":        2,
        },
        {
            "supplier_id":        "SUP008",
            "supplier_name":      "Pfizer USA",
            "country":            "United States",
            "region":             "North America",
            "lat": 40.713,        "lon": -74.006,
            "products_supplied":  ["Azithromycin_500mg", "Amoxicillin_250mg"],
            "delivery_delay_avg_days": 2.0,
            "on_time_delivery_pct":    93,
            "single_source":           False,
            "financial_stability":      10,
            "regulatory_compliance":    10,
            "geopolitical_risk":        2,
            "years_partnership":        9,
        },
    ]

    # Calculate composite risk score (0 = safest, 100 = highest risk)
    for s in suppliers:
        delay_risk       = min(s["delivery_delay_avg_days"] / 10 * 100, 100)
        reliability_risk = 100 - s["on_time_delivery_pct"]
        single_src_risk  = 20 if s["single_source"] else 0
        financial_risk   = (10 - s["financial_stability"]) * 10
        compliance_risk  = (10 - s["regulatory_compliance"]) * 10
        geo_risk         = s["geopolitical_risk"] * 10

        composite_risk = (
            delay_risk       * 0.25 +
            reliability_risk * 0.25 +
            single_src_risk  * 0.15 +
            financial_risk   * 0.15 +
            compliance_risk  * 0.10 +
            geo_risk         * 0.10
        )
        s["risk_score"]    = round(composite_risk, 1)
        s["risk_category"] = (
            "HIGH"   if composite_risk >= 50 else
            "MEDIUM" if composite_risk >= 25 else
            "LOW"
        )
        s["products_supplied"] = ", ".join(s["products_supplied"])

    df = pd.DataFrame(suppliers)
    path = f"{OUTPUT_DIR}/supplier_data.csv"
    df.to_csv(path, index=False)
    print(f"   ✅ Saved: {path}")
    print(f"   Suppliers: {len(df)} | High risk: {(df['risk_category']=='HIGH').sum()} | Low risk: {(df['risk_category']=='LOW').sum()}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 5 — KPI SUMMARY DATA
# Monthly aggregated KPIs for the dashboard summary panel
# ════════════════════════════════════════════════════════════

def generate_kpi_data(demand_df, inventory_df):
    print("\n[4/4] Generating monthly KPI summary data...")

    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_df["month"] = demand_df["date"].dt.to_period("M").astype(str)

    monthly = demand_df.groupby("month").agg(
        total_demand_units  = ("demand_units",     "sum"),
        total_demand_eur    = ("demand_value_eur",  "sum"),
        avg_daily_demand    = ("demand_units",      "mean"),
    ).reset_index()

    # Simulate AI vs traditional forecast error
    monthly["ai_forecast_error_pct"]          = np.random.uniform(3, 8,   len(monthly))
    monthly["traditional_forecast_error_pct"] = np.random.uniform(14, 28, len(monthly))
    monthly["forecast_improvement_pct"]       = (
        monthly["traditional_forecast_error_pct"] - monthly["ai_forecast_error_pct"]
    ).round(1)

    # Stockout and waste metrics from inventory
    inv_monthly = inventory_df.groupby("month").agg(
        stockout_events = ("status", lambda x: (x == "CRITICAL").sum()),
        reorder_events  = ("reorder_triggered", "sum"),
    ).reset_index()

    monthly = monthly.merge(inv_monthly, on="month", how="left")

    # Simulate waste and cost savings
    monthly["waste_units_traditional"] = (monthly["total_demand_units"] * np.random.uniform(0.06, 0.10, len(monthly))).astype(int)
    monthly["waste_units_ai"]          = (monthly["total_demand_units"] * np.random.uniform(0.01, 0.04, len(monthly))).astype(int)
    monthly["waste_reduction_pct"]     = (
        (monthly["waste_units_traditional"] - monthly["waste_units_ai"])
        / monthly["waste_units_traditional"] * 100
    ).round(1)

    monthly["cost_saving_eur"] = (
        (monthly["waste_units_traditional"] - monthly["waste_units_ai"]) * 0.15
    ).round(2)

    monthly["order_fulfillment_rate_pct"] = np.random.uniform(92, 99, len(monthly)).round(1)

    path = f"{OUTPUT_DIR}/kpi_data.csv"
    monthly.to_csv(path, index=False)
    print(f"   ✅ Saved: {path}")
    print(f"   Months: {len(monthly)} | Avg AI forecast error: {monthly['ai_forecast_error_pct'].mean():.1f}%")
    return monthly


# ════════════════════════════════════════════════════════════
# MAIN — Run all generators
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demand_df    = generate_demand_data()
    inventory_df = generate_inventory_data()
    supplier_df  = generate_supplier_data()
    kpi_df       = generate_kpi_data(demand_df, inventory_df)

    print("\n" + "=" * 60)
    print("  ✅ All 4 datasets generated successfully!")
    print("=" * 60)
    print(f"\n  📁 Files saved to: {OUTPUT_DIR}/")
    print("     demand_data.csv    — daily demand for 10 products")
    print("     inventory_data.csv — monthly stock levels & alerts")
    print("     supplier_data.csv  — 8 suppliers with risk scores")
    print("     kpi_data.csv       — monthly KPI summary")
    print("\n  Next step → run eda.py to explore the data")
    print("=" * 60)
