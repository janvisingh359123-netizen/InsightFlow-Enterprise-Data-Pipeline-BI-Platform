# Machine Learning Pipeline for SmartHouse AI
# Performs feature engineering, trains and compares 7 regression models,
# tracks metrics in MLflow, saves the best model, and computes SHAP importances.

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    print(f"XGBoost import disabled: {e}")
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception as e:
    print(f"LightGBM import disabled: {e}")
    HAS_LGB = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Add parent directory to path to import database adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import get_db_connection, DATABASE_TYPE

# Set MLflow tracking URI to SQLite database to comply with MLflow v3 standard
db_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(db_dir, "data"), exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(db_dir, 'data', 'mlflow.db')}")
mlflow.set_experiment("SmartHouse_AI_Valuation")

def load_data_from_db():
    """Fetches records from database and performs joins to construct raw training dataframe"""
    print("Fetching training data from database...")
    conn = get_db_connection()
    
    # Query to fetch latest sale price per property to train current valuation
    query = """
        WITH LatestSales AS (
            SELECT property_id, sale_price, sale_date,
                   ROW_NUMBER() OVER (PARTITION BY property_id ORDER BY sale_date DESC) as rn
            FROM historical_sales
        )
        SELECT p.*, l.city, l.state, l.zip_code,
               a.has_pool, a.has_garden, a.has_balcony, a.has_parking, a.has_security, a.has_gym,
               s.sale_price
        FROM properties p
        JOIN locations l ON p.location_id = l.location_id
        JOIN amenities a ON p.property_id = a.property_id
        JOIN LatestSales s ON p.property_id = s.property_id AND s.rn = 1
    """
    
    # Adapt SQL syntax for SQLite if needed (SQLite supports standard CTE + ROW_NUMBER in modern versions)
    # The sqlite3 version on Mac is modern and supports window functions.
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Successfully loaded {len(df)} property records.")
    return df

def feature_engineering(df):
    """Generates advanced features to boost model predictive capacity"""
    print("Performing feature engineering...")
    df = df.copy()
    
    # 1. Property Age
    df["property_age"] = 2026 - df["year_built"]
    
    # 2. Luxury Score (normalized sum of amenities)
    amenity_cols = ["has_pool", "has_garden", "has_balcony", "has_parking", "has_security", "has_gym"]
    df["luxury_score"] = df[amenity_cols].mean(axis=1)
    
    # 3. Accessibility Score (scaled 0 to 1, higher is better accessibility)
    # Weights: metro (40%), city center (40%), airport (20%)
    df["accessibility_score"] = np.clip(
        1.0 - (
            (df["distance_to_metro"] / 15.0) * 0.4 + 
            (df["distance_to_center"] / 30.0) * 0.4 + 
            (df["distance_to_airport"] / 50.0) * 0.2
        ), 0, 1
    )
    
    # 4. Neighborhood Score (combines school rating, crime rate, and luxury/infrastructure)
    df["neighborhood_score"] = (
        (df["school_rating"] / 10.0) * 0.5 + 
        ((10.0 - df["crime_rate"]) / 10.0) * 0.3 + 
        (df["luxury_score"] * 0.2)
    )
    
    # 5. Average ZIP-code price per square foot (calculated historically to avoid leakage)
    # Using average price per sqft grouped by zip_code
    zip_means = df.groupby("zip_code").apply(
        lambda x: (x["sale_price"] / x["area_sqft"]).mean()
    ).to_dict()
    df["zip_price_per_sqft"] = df["zip_code"].map(zip_means)
    
    return df, zip_means

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_evaluate_models():
    """Main training loop: preprocesses data, trains 7 regressors, and tracks with MLflow"""
    raw_df = load_data_from_db()
    
    # Feature Engineering
    df, zip_means = feature_engineering(raw_df)
    
    # Save ZIP-code averages for inference pipeline lookup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "ml", "zip_price_averages.json"), "w") as f:
        json.dump(zip_means, f, indent=4)
    
    # Define features and target
    features = [
        "property_type", "area_sqft", "bedrooms", "bathrooms", "floors", "garage_spaces",
        "property_age", "city", "crime_rate", "school_rating", "distance_to_center",
        "distance_to_metro", "distance_to_airport", "nearby_hospitals", "nearby_shopping",
        "has_pool", "has_garden", "has_balcony", "has_parking", "has_security", "has_gym",
        "luxury_score", "accessibility_score", "neighborhood_score", "zip_price_per_sqft"
    ]
    
    X = df[features]
    y = df["sale_price"]
    
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Column preprocessing
    num_features = [
        "area_sqft", "bedrooms", "bathrooms", "floors", "garage_spaces", "property_age",
        "crime_rate", "school_rating", "distance_to_center", "distance_to_metro",
        "distance_to_airport", "nearby_hospitals", "nearby_shopping",
        "luxury_score", "accessibility_score", "neighborhood_score", "zip_price_per_sqft"
    ]
    cat_features = ["property_type", "city"]
    passthrough_features = ["has_pool", "has_garden", "has_balcony", "has_parking", "has_security", "has_gym"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("pass", "passthrough", passthrough_features)
        ]
    )
    
    # Models dict
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
    
    comparison_results = []
    trained_pipelines = {}
    best_model_name = None
    best_r2 = -float("inf")
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Build scikit-learn pipeline (Preprocess -> Model)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # MLflow run logging
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = pipeline.predict(X_test)
            
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            mape = float(mean_absolute_percentage_error(y_test, y_pred))
            
            # Log metrics to MLflow
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAPE", mape)
            
            # Log hyperparameters (e.g. max_depth)
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)
                
            # Log the pipeline model to MLflow
            mlflow.sklearn.log_model(pipeline, "model")
            
            print(f"Finished {name}: MAE=${mae:,.2f}, RMSE=${rmse:,.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
            
            comparison_results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape
            })
            
            trained_pipelines[name] = pipeline
            
            # Identify the best performing model
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                
    # Save the model comparison results table as JSON
    comparison_df = pd.DataFrame(comparison_results)
    comparison_json_path = os.path.join(base_dir, "ml", "model_comparison.json")
    with open(comparison_json_path, "w") as f:
        json.dump(comparison_results, f, indent=4)
        
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model Selected: {best_model_name} (R2={best_r2:.4f})")
    
    # Save the best pipeline model locally for backend loading
    best_pipeline = trained_pipelines[best_model_name]
    models_dir = os.path.join(base_dir, "ml", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_pipeline, os.path.join(models_dir, "best_model.joblib"))
    print(f"Saved best model pipeline locally to {models_dir}/best_model.joblib")
    
    # Expose mean values for all features to compute local feature contributions in the API
    feature_means = X_train[num_features].mean().to_dict()
    pass_features_means = ["has_pool", "has_garden", "has_balcony", "has_parking", "has_security", "has_gym"]
    for c in pass_features_means:
        feature_means[c] = float(X_train[c].mean())
    # Save feature means
    with open(os.path.join(base_dir, "ml", "feature_means.json"), "w") as f:
        json.dump(feature_means, f, indent=4)
    
    # 7. Generate Explainable AI: Feature Importances
    # Extract feature importances from the best model if tree-based, otherwise use coefficients
    best_model = best_pipeline.named_steps["regressor"]
    
    # Map preprocessed features to identify actual names
    # Cat encoder categories
    cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols_encoded = list(cat_encoder.get_feature_names_out(cat_features))
    
    all_preprocessed_cols = num_features + cat_cols_encoded + passthrough_features
    
    importances_dict = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        # Normalize to sum to 100
        importances = (importances / importances.sum()) * 100
        for col, imp in zip(all_preprocessed_cols, importances):
            importances_dict[col] = float(imp)
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_)
        coefs = (coefs / coefs.sum()) * 100
        for col, coef in zip(all_preprocessed_cols, coefs):
            importances_dict[col] = float(coef)
    else:
        # Uniform fallback
        for col in all_preprocessed_cols:
            importances_dict[col] = 1.0
            
    # Group importances back into original feature sets to present in UI cleanly
    grouped_importances = {}
    for col, imp in importances_dict.items():
        original_feature = None
        for orig in features:
            if col.startswith(orig):
                original_feature = orig
                break
        if not original_feature:
            original_feature = col
            
        grouped_importances[original_feature] = grouped_importances.get(original_feature, 0.0) + imp
        
    # Order descending
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)
    sorted_importances_dict = [{"feature": k, "importance": round(v, 2)} for k, v in sorted_importances]
    
    with open(os.path.join(base_dir, "ml", "feature_importances.json"), "w") as f:
        json.dump(sorted_importances_dict, f, indent=4)
        
    print("Feature importances computed and saved successfully.")

if __name__ == "__main__":
    train_and_evaluate_models()
