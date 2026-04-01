# EcoPack Flask – AI-Powered Sustainable Packaging Recommender

EcoPack Flask is a Flask-based backend and lightweight UI for recommending packaging materials that balance cost and environmental impact. It uses trained ML models (Random Forest and XGBoost) on an engineered materials dataset to:

- Predict material cost and CO₂ emission scores
- Rank candidate materials by a combined “AI Recommendation Score”
- Provide a web UI for interactive recommendations
- Offer a sustainability analytics dashboard with Plotly charts
- Persist product definitions in PostgreSQL for reuse

## Features

- **AI material recommendation**
  - Inputs: product weight and fragility level
  - Outputs: top-N materials with predicted cost, CO₂, recyclability, biodegradability, and AI score
- **Sustainability dashboard**
  - CO₂ reduction vs. baseline
  - Cost savings vs. baseline
  - Average recyclability and biodegradability
  - Material usage trends
- **PostgreSQL integration**
  - `materials` table for material properties
  - `products` table for saved product configurations
- **REST APIs**
  - Health check
  - Product creation
  - Recommendation endpoint(s) used by the UI and available to other clients

## Project Structure

- app.py – Flask application, API routes, SQLAlchemy models, Plotly dashboard logic
- train_models.py – model training pipeline; produces `.joblib` models and ranking CSV
- engineered_dataset.csv – engineered training dataset
- AI_Material_Ranking.csv – ranked materials exported by training script
- rf_cost_pipeline.joblib – Random Forest pipeline for cost prediction
- xgb_co2_pipeline.joblib – XGBoost pipeline for CO₂ prediction
- templates/base.html – base layout (Bootstrap, shared styling)
- templates/materials_ui.html – AI material recommendation UI
- templates/dashboard.html – sustainability analytics dashboard
- requirements.txt – Python dependencies
- Procfile – process definition for platforms like Heroku/Render
- docs/TECHNICAL.md – detailed technical documentation

## Prerequisites

- OS: Windows (tested) or any OS with Python
- Python: 3.x (compatible with Flask, scikit-learn, xgboost)
- PostgreSQL instance accessible via `DATABASE_URL`
- Recommended: virtual environment (`venv`)

## Setup

1. **Create and activate a virtual environment**

   ```bash
   cd ecopack-flask
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   # source .venv/bin/activate  # on macOS/Linux
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure PostgreSQL**

   Create a database (e.g., `ecopack`) and tables `materials` and `products` aligned with the ORM models in `app.py`.

   Then set the `DATABASE_URL` environment variable, for example in PowerShell:

   ```powershell
   $env:DATABASE_URL = "postgresql://username:password@localhost:5432/ecopack"
   ```

   If `DATABASE_URL` is not set, `app.py` will fall back to a default Neon connection string (intended for development/demo).

4. **Prepare the dataset**

   Ensure `engineered_dataset.csv` exists in the project root. This is the dataset used for training and for dashboard metrics.

5. **Train and save models**

   Run once to train and persist models:

   ```bash
   python train_models.py
   ```

   This generates:

   - `rf_cost_pipeline.joblib`
   - `xgb_co2_pipeline.joblib`
   - `AI_Material_Ranking.csv`

## Running the Application

### Local development (Flask)

```bash
python app.py
```

By default, the app runs on `http://127.0.0.1:5000/` (or as configured in `app.py`).

Open:

- `http://127.0.0.1:5000/ui` – AI material recommendation UI
- `http://127.0.0.1:5000/dashboard` – sustainability analytics dashboard
- `http://127.0.0.1:5000/api/health` – health check JSON

### Production / WSGI (gunicorn + Procfile)

On platforms like Heroku/Render:

- `Procfile` defines how to start the app with `gunicorn`.
- Ensure `DATABASE_URL` is set in the platform’s environment.

## Key Endpoints

- `GET /api/health`
  - Returns JSON with `database_connected`, `models_loaded`, `data_loaded`.
- `POST /api/product`
  - JSON body:
    ```json
    {
      "product_name": "Glass Bottle",
      "category": "food",
      "product_weight_kg": 1.5,
      "fragility_level": "High"
    }
    ```
  - Persists the product to the `products` table when the DB is available.
- `/ui`
  - Renders the material recommendation form and results table.
- `/dashboard`
  - Renders the sustainability dashboard with Plotly charts.

## Tech Stack

- Backend: Flask, Flask-CORS
- Data / ML: pandas, numpy, scikit-learn, xgboost, joblib
- Database: PostgreSQL via SQLAlchemy
- Visualization: Plotly (plotly.express, JSON serialization)
- Export: openpyxl, xlsxwriter, reportlab (for Excel/PDF exports)
- Deployment: gunicorn, Procfile-based platforms

## Development Tips

- Re-run `train_models.py` whenever:
  - You update `engineered_dataset.csv`
  - You change model hyperparameters or feature engineering
- Use `/api/health` to quickly confirm:
  - DB connectivity
  - Model files successfully loaded
  - Dataset is available
- Start with local DB; switch to managed PostgreSQL in production by updating `DATABASE_URL`.
