# -*- coding: utf-8 -*-
"""
Train and save ML models for EcoPack project (Milestone 2 output, used in Milestone 3).

- Reads engineered_dataset.csv from the current folder
- Trains:
  * RandomForestRegressor for Cost_per_KG_USD
  * XGBRegressor for CO2_Emission_Score_%
- Evaluates models (RMSE, MAE, R2)
- Computes AI_Recommendation_Score for all rows
- Saves:
  * rf_cost_pipeline.joblib
  * xgb_co2_pipeline.joblib
  * AI_Material_Ranking.csv

Run once before starting the Flask backend:
    python train_models.py
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ------------------------------
# Load engineered dataset
# ------------------------------

DATA_PATH = "engineered_dataset.csv"

engineered_df = pd.read_csv(DATA_PATH)

features = [
    "Material_Type",
    "Strength_PSI",
    "Weight_Capacity_KG",
    "Biodegradability_%",
    "Recyclability_%",
]

X = engineered_df[features]

# Targets
y_cost = engineered_df["Cost_per_KG_USD"]
y_co2 = engineered_df["CO2_Emission_Score_%"]

# ------------------------------
# Train-test split
# ------------------------------

X_train, X_test, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)

# ------------------------------
# Preprocessing
# ------------------------------

categorical_features = ["Material_Type"]

numerical_features = [
    "Strength_PSI",
    "Weight_Capacity_KG",
    "Biodegradability_%",
    "Recyclability_%",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ------------------------------
# Random Forest for Cost
# ------------------------------

rf_cost_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
            ),
        ),
    ]
)

rf_cost_pipeline.fit(X_train, y_cost_train)

rf_cost_predictions = rf_cost_pipeline.predict(X_test)

rf_cost_rmse = np.sqrt(mean_squared_error(y_cost_test, rf_cost_predictions))
rf_cost_mae = mean_absolute_error(y_cost_test, rf_cost_predictions)
rf_cost_r2 = r2_score(y_cost_test, rf_cost_predictions)

print("===== RANDOM FOREST (COST) =====")
print("RMSE:", rf_cost_rmse)
print("MAE:", rf_cost_mae)
print("R2 Score:", rf_cost_r2)

# ------------------------------
# XGBoost for CO2
# ------------------------------

xgb_co2_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
            ),
        ),
    ]
)

xgb_co2_pipeline.fit(X_train, y_co2_train)

xgb_co2_predictions = xgb_co2_pipeline.predict(X_test)

xgb_co2_rmse = np.sqrt(mean_squared_error(y_co2_test, xgb_co2_predictions))
xgb_co2_mae = mean_absolute_error(y_co2_test, xgb_co2_predictions)
xgb_co2_r2 = r2_score(y_co2_test, xgb_co2_predictions)

print("\n===== XGBOOST (CO2) =====")
print("RMSE:", xgb_co2_rmse)
print("MAE:", xgb_co2_mae)
print("R2 Score:", xgb_co2_r2)

# ------------------------------
# AI Recommendation System
# ------------------------------

# Predict cost and CO2 for entire dataset
engineered_df["Predicted_Cost"] = rf_cost_pipeline.predict(X)
engineered_df["Predicted_CO2"] = xgb_co2_pipeline.predict(X)

# Normalize predictions (lower is better)
engineered_df["Cost_Score"] = 1 - (
    (engineered_df["Predicted_Cost"] - engineered_df["Predicted_Cost"].min())
    / (engineered_df["Predicted_Cost"].max() - engineered_df["Predicted_Cost"].min())
)

engineered_df["CO2_Score"] = 1 - (
    (engineered_df["Predicted_CO2"] - engineered_df["Predicted_CO2"].min())
    / (engineered_df["Predicted_CO2"].max() - engineered_df["Predicted_CO2"].min())
)

engineered_df["AI_Recommendation_Score"] = (
    engineered_df["Cost_Score"] * 0.5 + engineered_df["CO2_Score"] * 0.5
)

ranked_materials = engineered_df.sort_values(
    by="AI_Recommendation_Score", ascending=False
)

print("\n===== TOP 10 RECOMMENDED MATERIALS (TRAIN SCRIPT) =====")
print(
    ranked_materials[
        [
            "Material_Type",
            "Predicted_Cost",
            "Predicted_CO2",
            "AI_Recommendation_Score",
        ]
    ].head(10)
)

# Save ranked output
ranked_materials.to_csv("AI_Material_Ranking.csv", index=False)

# ------------------------------
# Save models
# ------------------------------

rf_cost_model_filename = "rf_cost_pipeline.joblib"
xgb_co2_model_filename = "xgb_co2_pipeline.joblib"

joblib.dump(rf_cost_pipeline, rf_cost_model_filename)
print(f"Random Forest Cost Model saved to {rf_cost_model_filename}")

joblib.dump(xgb_co2_pipeline, xgb_co2_model_filename)
print(f"XGBoost CO2 Model saved to {xgb_co2_model_filename}")

print("\n✅ Training complete. Models and ranking file saved.")
