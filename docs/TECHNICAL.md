# EcoPack Flask – Technical Documentation

This document describes the internal architecture of the EcoPack Flask project, including the ML pipeline, recommendation logic, database layer, and dashboards.

## 1. Machine Learning Pipeline (train_models.py)

### 1.1 Data and Features

- Source: `engineered_dataset.csv`
- Feature columns:
  - `Material_Type` (categorical)
  - `Strength_PSI`
  - `Weight_Capacity_KG`
  - `Biodegradability_%`
  - `Recyclability_%`
- Target columns:
  - `Cost_per_KG_USD`
  - `CO2_Emission_Score_%`

### 1.2 Preprocessing

- Uses `ColumnTransformer` with:
  - Numerical features (passthrough):
    - `Strength_PSI`, `Weight_Capacity_KG`, `Biodegradability_%`, `Recyclability_%`
  - Categorical features:
    - `Material_Type` → `OneHotEncoder(handle_unknown="ignore")`

This preprocessor is shared between both models via scikit-learn `Pipeline`s.

### 1.3 Models

- **Cost model**
  - Algorithm: `RandomForestRegressor`
  - Hyperparameters (key):
    - `n_estimators=200`
    - `max_depth=10`
    - `random_state=42`
- **CO₂ model**
  - Algorithm: `XGBRegressor`
  - Hyperparameters (key):
    - `n_estimators=200`
    - `learning_rate=0.1`
    - `max_depth=6`
    - `random_state=42`

Each model is wrapped in a scikit-learn `Pipeline` with the shared preprocessor and trained on an 80/20 train-test split.

### 1.4 Evaluation

For both targets, the script computes:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² score

Metrics are printed to stdout during training for quick inspection.

### 1.5 AI Recommendation Score and Outputs

After training, the script:

1. Predicts `Predicted_Cost` and `Predicted_CO2` for all rows in `engineered_df`.
2. Normalizes predictions so that lower raw values become higher scores:
   - `Cost_Score` = `1 - normalized(Predicted_Cost)`
   - `CO2_Score` = `1 - normalized(Predicted_CO2)`
3. Aggregates into `AI_Recommendation_Score`:
   - `AI_Recommendation_Score = 0.5 * Cost_Score + 0.5 * CO2_Score`
4. Sorts materials by `AI_Recommendation_Score` (descending) and writes the full ranking to `AI_Material_Ranking.csv`.
5. Saves the trained pipelines as:
   - `rf_cost_pipeline.joblib`
   - `xgb_co2_pipeline.joblib`

These `.joblib` files are consumed at runtime by the Flask app.

---

## 2. Runtime Logic (app.py)

### 2.1 Model and Data Loading

Constants:

- `MODEL_COST_PATH = "rf_cost_pipeline.joblib"`
- `MODEL_CO2_PATH = "xgb_co2_pipeline.joblib"`
- `DATA_PATH = "engineered_dataset.csv"`

On import, the app attempts to:

- Load `rf_model` from `MODEL_COST_PATH`
- Load `xgb_model` from `MODEL_CO2_PATH`
- Load `engineered_df` from `DATA_PATH`

If loading fails, it logs an error and sets the corresponding variables to `None`. The `/api/health` endpoint exposes these statuses.

### 2.2 Environment Scoring

Function: `compute_environment_scores(cost_values, co2_values)`

- Inputs: 1D numpy arrays `cost_values` and `co2_values`.
- Steps:
  - Compute `cost_min`, `cost_max`, `co2_min`, `co2_max`.
  - Derive ranges, with a fallback of `1.0` to avoid division by zero.
  - Normalize and invert:
    - `cost_scores = 1 - (cost - cost_min) / (cost_max - cost_min)`
    - `co2_scores = 1 - (co2 - co2_min) / (co2_max - co2_min)`
  - Compute `ai_scores = 0.5 * cost_scores + 0.5 * co2_scores`.
- Returns: `cost_scores`, `co2_scores`, `ai_scores` as numpy arrays.

This logic is central to both offline ranking and online recommendations.

### 2.3 Recommendation Generation

Function: `generate_material_recommendations(product_weight, fragility_level, top_n=5)`

- Validates that `rf_model`, `xgb_model`, and `engineered_df` are not `None`.
- Maps `fragility_level` → minimum required strength using `FRAGILITY_STRENGTH_MAP`:
  - `low`: 1000 PSI
  - `medium`: 3000 PSI
  - `high`: 5000 PSI
  - `very_high`: 8000 PSI
- Cleans `top_n` to an integer between 1 and 50.
- Filters candidate materials from `engineered_df`:
  - `Weight_Capacity_KG >= product_weight`
  - `Strength_PSI >= min_strength`
- If no rows remain, returns an object with `total_candidates = 0` and an empty `recommendations` list.
- For candidates:
  - Builds `X_candidates = df_filtered[features]`.
  - Calls `rf_model.predict` and `xgb_model.predict`.
  - Adds `Predicted_Cost` and `Predicted_CO2` columns.
  - Calls `compute_environment_scores` to derive `Cost_Score`, `CO2_Score`, and `AI_Recommendation_Score`.
  - Sorts by `AI_Recommendation_Score` (descending) and truncates to `top_n`.
- Returns a dict with:
  - `total_candidates`
  - `top_n`
  - `product_constraints` (weight, fragility, min_strength_psi)
  - `recommendations`: list of per-material dicts containing raw attributes, predictions, and scores.

This function is reused by both the API layer and the UI route for consistency.

---

## 3. Database Layer

### 3.1 Configuration and Engine

- `DATABASE_URL` is read from the environment, with a default Neon connection string as fallback.
- `init_db()` creates:
  - `engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})`
  - `SessionLocal = sessionmaker(bind=engine)`
- `DB_AVAILABLE` is set based on whether `init_db()` succeeds.

### 3.2 ORM Models

Defined via SQLAlchemy `declarative_base()`:

- **Material** (`materials` table)
  - `material_id` (Integer, primary key)
  - `material_type` (String)
  - `strength_psi` (Float)
  - `weight_capacity_kg` (Float)
  - `biodegradability_percent` (Float)
  - `co2_emission_score_percent` (Float)
  - `recyclability_percent` (Float)
  - `cost_per_kg_usd` (Float)

- **Product** (`products` table)
  - `product_id` (Integer, primary key)
  - `product_name` (String)
  - `category` (String)
  - `product_weight_kg` (Float)
  - `fragility_level` (String)

### 3.3 Product API

Route: `POST /api/product`

- Expects JSON body with:
  - `product_name`
  - `category`
  - `product_weight_kg`
  - `fragility_level`
- Behavior:
  - If `DB_AVAILABLE` or `SessionLocal` is `False`, returns HTTP 503 with an error message.
  - Otherwise, opens a DB session via `SessionLocal`, creates a `Product` instance, adds and commits it.

This endpoint is called from the front-end form in `materials_ui.html` using Fetch.

---

## 4. Dashboard Analytics

### 4.1 Metric Computation

Function: `compute_dashboard_metrics(df)`

- Requires columns:
  - `CO2_Emission_Score_%`
  - `Cost_per_KG_USD`
  - `Material_Suitability_Score`
- Steps:
  - Compute overall averages: `overall_co2`, `overall_cost`.
  - Derive `top_df` as the top 10 rows by `Material_Suitability_Score`.
  - Compute `top_co2`, `top_cost` from `top_df`.
  - Compute KPIs:
    - `co2_reduction_pct = (overall_co2 - top_co2) / overall_co2 * 100`
    - `cost_savings_pct = (overall_cost - top_cost) / overall_cost * 100`
  - Optionally compute:
    - `avg_recyclability` from `Recyclability_%` (if present)
    - `avg_biodegradability` from `Biodegradability_%` (if present)

These metrics are passed to the dashboard template and displayed as KPI cards.

### 4.2 Plotly Figures

Function: `build_dashboard_figures(df)`

- Builds three aggregated views:
  - `co2_by_material`: mean `CO2_Emission_Score_%` by `Material_Type`
  - `cost_by_material`: mean `Cost_per_KG_USD` by `Material_Type`
  - `usage_trend`: count of rows per `Material_Type`
- Creates three bar charts with `plotly.express.bar`:
  - CO₂ by material
  - Cost by material
  - Usage (frequency) by material
- Serializes figures via `json.dumps(figures, cls=PlotlyJSONEncoder)` and injects them as `graphs_json` into the `dashboard.html` template.

On the client, Plotly JS reconstructs the charts with `Plotly.newPlot`.

---

## 5. UI Integration

### 5.1 Recommendation UI (templates/materials_ui.html)

- Primary form posts to `/ui` with fields:
  - `product_weight_kg`
  - `fragility_level`
  - `top_n`
- Backend route:
  - Parses form data.
  - Calls `generate_material_recommendations`.
  - Renders the template with `result` and `fragility_options`.
- Template behavior:
  - If `result.recommendations` exists, it shows:
    - Summary cards (total candidates, top shown, min strength).
    - A table of recommended materials with raw attributes, predictions, and AI score.
  - If no results, shows contextual alerts.

### 5.2 Product Save UI

- Secondary form (`#product-form`) in `materials_ui.html` sends a JSON POST to `/api/product` using Fetch.
- Displays inline messages in `#product-message` based on the API response.

### 5.3 Dashboard UI (templates/dashboard.html)

- Expects:
  - `metrics` dict from `compute_dashboard_metrics`.
  - `graphs_json` from `build_dashboard_figures`.
- Renders KPI cards for:
  - CO₂ reduction (%)
  - Cost savings (%)
  - Average recyclability
  - Average biodegradability
- Plots charts:
  - CO₂ by material
  - Cost by material
  - Usage trends

---

## 6. Extensibility Notes

- To add new features or signals to the recommendation score:
  - Extend `engineered_dataset.csv` with new columns.
  - Update `features` in both `train_models.py` and `app.py`.
  - Adjust `compute_environment_scores` or the AI score weighting if needed.
- To change the persistence layer:
  - Replace the SQLAlchemy engine URL (e.g., to a different PostgreSQL instance).
  - Migrate the `materials` and `products` schemas accordingly.
- To integrate with external systems:
  - Expose additional REST endpoints that call `generate_material_recommendations` and return JSON responses.
