# Time Series Forecasting Module for SmartHouse AI
# Aggregates historical sale transactions by month,
# fits a trend + seasonality regression forecaster for each city and the global market,
# and generates price projections for the next 3, 6, and 12 months.

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# Add parent directory to path to import database adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import get_db_connection

def forecast_market_trends():
    """Aggregates sales by month, fits a trend + seasonal model, and saves forecasts to JSON"""
    print("Fetching historical transaction history for forecasting...")
    conn = get_db_connection()
    
    # Query historical sales joined with location to group by month and city
    query = """
        SELECT hs.sale_price, hs.sale_date, l.city
        FROM historical_sales hs
        JOIN properties p ON hs.property_id = p.property_id
        JOIN locations l ON p.location_id = l.location_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} historical sales records.")
    
    # Preprocess dates
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["year_month"] = df["sale_date"].dt.to_period("M")
    
    # Get lists of unique months and cities
    all_months = sorted(df["year_month"].unique())
    cities = sorted(df["city"].unique())
    
    # Aggregate data: monthly average prices globally and per city
    monthly_data = []
    
    for month in all_months:
        month_df = df[df["year_month"] == month]
        row = {"month_str": str(month), "global_avg": float(month_df["sale_price"].mean())}
        
        for city in cities:
            city_month_df = month_df[month_df["city"] == city]
            row[city] = float(city_month_df["sale_price"].mean()) if len(city_month_df) > 0 else np.nan
            
        monthly_data.append(row)
        
    ts_df = pd.DataFrame(monthly_data)
    
    # Interpolate missing values if any
    ts_df = ts_df.ffill().bfill()
    
    # Prepare forecasting models
    # We will forecast the next 12 months
    forecast_steps = 12
    
    # Generate future month strings
    last_month_period = all_months[-1]
    future_months = []
    for i in range(1, forecast_steps + 1):
        future_period = last_month_period + i
        future_months.append(str(future_period))
        
    # Build time features for modeling (Trend Index and Cyclic Fourier Features for Month)
    ts_df["trend"] = np.arange(len(ts_df)) + 1
    
    # Extrapolate calendar month numbers
    ts_df["month_num"] = pd.to_datetime(ts_df["month_str"]).dt.month
    ts_df["month_sin"] = np.sin(2 * np.pi * ts_df["month_num"] / 12)
    ts_df["month_cos"] = np.cos(2 * np.pi * ts_df["month_num"] / 12)
    
    # Future features
    future_trends = np.arange(len(ts_df), len(ts_df) + forecast_steps) + 1
    future_months_nums = [(last_month_period + i).month for i in range(1, forecast_steps + 1)]
    future_sin = np.sin(2 * np.pi * np.array(future_months_nums) / 12)
    future_cos = np.cos(2 * np.pi * np.array(future_months_nums) / 12)
    
    X_train = ts_df[["trend", "month_sin", "month_cos"]]
    X_forecast = pd.DataFrame({
        "trend": future_trends,
        "month_sin": future_sin,
        "month_cos": future_cos
    })
    
    # Outputs dictionaries
    historical_out = {
        "months": ts_df["month_str"].tolist(),
        "global_avg": [round(x, 2) for x in ts_df["global_avg"].tolist()]
    }
    
    forecast_out = {
        "months": future_months,
        "global_avg": []
    }
    
    for city in cities:
        historical_out[city] = [round(x, 2) for x in ts_df[city].tolist()]
        forecast_out[city] = []
        
    # Fit & Forecast for Global average
    model_global = Ridge(alpha=1.0)
    model_global.fit(X_train, ts_df["global_avg"])
    global_forecasts = model_global.predict(X_forecast)
    forecast_out["global_avg"] = [round(max(50000.0, x), 2) for x in global_forecasts.tolist()]
    
    # Fit & Forecast for each City
    for city in cities:
        model_city = Ridge(alpha=1.0)
        model_city.fit(X_train, ts_df[city])
        city_forecasts = model_city.predict(X_forecast)
        forecast_out[city] = [round(max(50000.0, x), 2) for x in city_forecasts.tolist()]
        
    output_payload = {
        "historical": historical_out,
        "forecast": forecast_out
    }
    
    # Save output to JSON
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, "ml", "forecast_results.json")
    with open(output_file, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"Time series forecasts generated successfully and written to {output_file}.")

if __name__ == "__main__":
    forecast_market_trends()
