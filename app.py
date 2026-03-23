# -*- coding: utf-8 -*-
"""
EcoPack Flask Backend (Milestone 3 - Module 5)

Provides REST APIs for:
- Product input handling
- AI material recommendation
- Environmental score computation

Also connects to PostgreSQL (materials, products tables) using SQLAlchemy.

Before running:
1) Ensure dependencies are installed in your virtualenv:
   pip install flask flask-cors sqlalchemy psycopg2-binary pandas numpy joblib xgboost scikit-learn

2) Train and save models by running (once):
   python train_models.py

3) Set DATABASE_URL environment variable for PostgreSQL, e.g. (Windows PowerShell):
   $env:DATABASE_URL = "postgresql://username:password@localhost:5432/ecopack"

Then start the server:
   python app.py
"""

import os
import math
import io
import json

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# ================================
# Flask app setup
# ================================

app = Flask(__name__)
CORS(app)  # allow cross-origin for frontend

# ================================
# Database (PostgreSQL via SQLAlchemy)
# ================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # Placeholder; override with real credentials via env var
    "postgresql://postgres:Ajay7624@localhost:5432/ecopack",
)

Base = declarative_base()


class Material(Base):
    """Map to existing materials table in PostgreSQL"""

    __tablename__ = "materials"

    material_id = Column(Integer, primary_key=True)
    material_type = Column(String(100))
    strength_psi = Column(Float)
    weight_capacity_kg = Column(Float)
    biodegradability_percent = Column(Float)
    co2_emission_score_percent = Column(Float)
    recyclability_percent = Column(Float)
    cost_per_kg_usd = Column(Float)


class Product(Base):
    """Map to existing products table in PostgreSQL"""

    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(100))
    category = Column(String(50))
    product_weight_kg = Column(Float)
    fragility_level = Column(String(20))


# Create engine and session (if DB is reachable)
engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal
    try:
        engine = create_engine(DATABASE_URL, echo=False)
        SessionLocal = sessionmaker(bind=engine)
        # Do NOT create tables; they are already defined in PostgreSQL
        # Base.metadata.create_all(engine)
        return True
    except Exception as exc:
        # Log to console only; API will report DB status via /api/health
        print(f"[DB] Connection failed: {exc}")
        engine = None
        SessionLocal = None
        return False


DB_AVAILABLE = init_db()

# ================================
# Load models and data
# ================================

MODEL_COST_PATH = "rf_cost_pipeline.joblib"
MODEL_CO2_PATH = "xgb_co2_pipeline.joblib"
DATA_PATH = "engineered_dataset.csv"

rf_model = None
xgb_model = None
engineered_df = None

features = [
    "Material_Type",
    "Strength_PSI",
    "Weight_Capacity_KG",
    "Biodegradability_%",
    "Recyclability_%",
]


def load_models_and_data():
    global rf_model, xgb_model, engineered_df
    try:
        rf_model = joblib.load(MODEL_COST_PATH)
        xgb_model = joblib.load(MODEL_CO2_PATH)
    except Exception as exc:
        print(f"[MODEL] Error loading models: {exc}")
        rf_model = None
        xgb_model = None

    try:
        engineered_df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        print(f"[DATA] Error loading dataset: {exc}")
        engineered_df = None


load_models_and_data()


# ================================
# Helper functions
# ================================

def compute_environment_scores(cost_values, co2_values):
    """Compute normalized cost/CO2 scores and AI score for arrays.

    cost_values, co2_values: 1D numpy arrays.
    Returns cost_scores, co2_scores, ai_scores (numpy arrays).
    """

    cost_min = float(cost_values.min())
    cost_max = float(cost_values.max())
    co2_min = float(co2_values.min())
    co2_max = float(co2_values.max())

    # Avoid division by zero
    cost_range = cost_max - cost_min or 1.0
    co2_range = co2_max - co2_min or 1.0

    cost_scores = 1.0 - (cost_values - cost_min) / cost_range
    co2_scores = 1.0 - (co2_values - co2_min) / co2_range
    ai_scores = 0.5 * cost_scores + 0.5 * co2_scores
    return cost_scores, co2_scores, ai_scores


FRAGILITY_STRENGTH_MAP = {
    "low": 1000,
    "medium": 3000,
    "high": 5000,
    "very_high": 8000,
}


def generate_material_recommendations(product_weight, fragility_level, top_n=5):
    """Core recommendation logic used by both API and UI.

    Returns a dict with keys:
        total_candidates, top_n, product_constraints, recommendations
    """

    if rf_model is None or xgb_model is None or engineered_df is None:
        raise RuntimeError("Models or dataset not loaded")

    fragility_raw = str(fragility_level).strip().lower().replace(" ", "_")
    min_strength = FRAGILITY_STRENGTH_MAP.get(fragility_raw, 3000)

    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = 5
    top_n = max(1, min(top_n, 50))

    # Filter materials based on simple business rules
    df = engineered_df.copy()
    df_filtered = df[
        (df["Weight_Capacity_KG"] >= product_weight)
        & (df["Strength_PSI"] >= min_strength)
    ]

    if df_filtered.empty:
        return {
            "total_candidates": 0,
            "top_n": 0,
            "product_constraints": {
                "product_weight_kg": float(product_weight),
                "fragility_level": fragility_raw,
                "min_strength_psi": int(min_strength),
            },
            "recommendations": [],
        }

    # Predict cost and CO2 for filtered materials
    X_candidates = df_filtered[features]
    pred_cost = rf_model.predict(X_candidates)
    pred_co2 = xgb_model.predict(X_candidates)

    df_filtered = df_filtered.copy()
    df_filtered["Predicted_Cost"] = pred_cost
    df_filtered["Predicted_CO2"] = pred_co2

    # Compute normalized scores and AI score
    cost_scores, co2_scores, ai_scores = compute_environment_scores(pred_cost, pred_co2)

    df_filtered["Cost_Score"] = cost_scores
    df_filtered["CO2_Score"] = co2_scores
    df_filtered["AI_Recommendation_Score"] = ai_scores

    # Rank materials
    df_ranked = df_filtered.sort_values(
        by="AI_Recommendation_Score", ascending=False
    ).head(top_n)

    recommendations = []
    for _, row in df_ranked.iterrows():
        recommendations.append(
            {
                "material_type": row["Material_Type"],
                "strength_psi": float(row["Strength_PSI"]),
                "weight_capacity_kg": float(row["Weight_Capacity_KG"]),
                "biodegradability_percent": float(row["Biodegradability_%"]),
                "recyclability_percent": float(row["Recyclability_%"]),
                "cost_per_kg_usd": float(row["Cost_per_KG_USD"]),
                "predicted_cost": float(row["Predicted_Cost"]),
                "predicted_co2": float(row["Predicted_CO2"]),
                "cost_score": float(row["Cost_Score"]),
                "co2_score": float(row["CO2_Score"]),
                "ai_recommendation_score": float(row["AI_Recommendation_Score"]),
            }
        )

    return {
        "total_candidates": int(df_filtered.shape[0]),
        "top_n": len(recommendations),
        "product_constraints": {
            "product_weight_kg": float(product_weight),
            "fragility_level": fragility_raw,
            "min_strength_psi": int(min_strength),
        },
        "recommendations": recommendations,
    }


def compute_dashboard_metrics(df: pd.DataFrame):
    """Compute high-level sustainability metrics for the BI dashboard.

    Returns a dict with co2_reduction_pct, cost_savings_pct and
    some supporting averages used for reporting.
    """

    if df is None or df.empty:
        return None

    if not {"CO2_Emission_Score_%", "Cost_per_KG_USD", "Material_Suitability_Score"}.issubset(df.columns):
        return None

    overall_co2 = float(df["CO2_Emission_Score_%"].mean())
    overall_cost = float(df["Cost_per_KG_USD"].mean())

    top_df = df.sort_values("Material_Suitability_Score", ascending=False).head(10)
    top_co2 = float(top_df["CO2_Emission_Score_%"].mean())
    top_cost = float(top_df["Cost_per_KG_USD"].mean())

    co2_reduction_pct = (overall_co2 - top_co2) / overall_co2 * 100 if overall_co2 else 0.0
    cost_savings_pct = (overall_cost - top_cost) / overall_cost * 100 if overall_cost else 0.0

    avg_recyclability = float(df["Recyclability_%"].mean()) if "Recyclability_%" in df.columns else None
    avg_biodegradability = float(df["Biodegradability_%"].mean()) if "Biodegradability_%" in df.columns else None

    return {
        "co2_reduction_pct": co2_reduction_pct,
        "cost_savings_pct": cost_savings_pct,
        "overall_co2": overall_co2,
        "overall_cost": overall_cost,
        "top_co2": top_co2,
        "top_cost": top_cost,
        "avg_recyclability": avg_recyclability,
        "avg_biodegradability": avg_biodegradability,
    }


def build_dashboard_figures(df: pd.DataFrame):
    """Create Plotly figures for dashboard charts and return JSON."""

    if df is None or df.empty:
        return None

    # Aggregate by material type
    co2_by_material = (
        df.groupby("Material_Type")["CO2_Emission_Score_%"].mean().reset_index()
    )
    cost_by_material = (
        df.groupby("Material_Type")["Cost_per_KG_USD"].mean().reset_index()
    )
    usage_trend = df.groupby("Material_Type").size().reset_index(name="Usage_Count")

    fig_co2 = px.bar(
        co2_by_material,
        x="Material_Type",
        y="CO2_Emission_Score_%",
        title="Average CO2 Emission Score by Material",
        color="CO2_Emission_Score_%",
        color_continuous_scale="Greens_r",
    )
    fig_co2.update_layout(xaxis_title="Material Type", yaxis_title="CO2 Emission Score (%)")

    fig_cost = px.bar(
        cost_by_material,
        x="Material_Type",
        y="Cost_per_KG_USD",
        title="Average Cost per KG by Material",
        color="Cost_per_KG_USD",
        color_continuous_scale="Blues",
    )
    fig_cost.update_layout(xaxis_title="Material Type", yaxis_title="Cost per KG (USD)")

    fig_usage = px.bar(
        usage_trend,
        x="Material_Type",
        y="Usage_Count",
        title="Material Usage Trends (Dataset Frequency)",
        color="Usage_Count",
        color_continuous_scale="Purples",
    )
    fig_usage.update_layout(xaxis_title="Material Type", yaxis_title="Count in Dataset")

    figures = [fig_co2, fig_cost, fig_usage]
    return json.dumps(figures, cls=PlotlyJSONEncoder)


# ================================
# API routes
# ================================


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check: models, data, and DB status."""

    db_ok = DB_AVAILABLE and engine is not None
    models_ok = rf_model is not None and xgb_model is not None
    data_ok = engineered_df is not None

    return jsonify(
        {
            "success": True,
            "database_connected": db_ok,
            "models_loaded": models_ok,
            "data_loaded": data_ok,
        }
    )


@app.route("/api/product", methods=["POST"])
def create_product():
    """Product input handling: save product info into PostgreSQL products table.

    Expected JSON body:
    {
        "product_name": "Glass Bottle",
        "category": "food",
        "product_weight_kg": 1.5,
        "fragility_level": "High"  // Low / Medium / High / Very High
    }
    """

    if not DB_AVAILABLE or SessionLocal is None:
        return (
            jsonify({"success": False, "error": "Database not configured or not reachable"}),
            503,
        )

    data = request.get_json(silent=True) or {}

    required = ["product_name", "category", "product_weight_kg", "fragility_level"]
    for field in required:
        if field not in data:
            return (
                jsonify({"success": False, "error": f"Missing field: {field}"}),
                400,
            )

    try:
        product_weight = float(data["product_weight_kg"])
    except (TypeError, ValueError):
        return (
            jsonify({"success": False, "error": "product_weight_kg must be numeric"}),
            400,
        )

    fragility = str(data["fragility_level"]).strip()

    session = SessionLocal()
    try:
        product = Product(
            product_name=data["product_name"],
            category=data["category"],
            product_weight_kg=product_weight,
            fragility_level=fragility,
        )
        session.add(product)
        session.commit()
        session.refresh(product)

        return (
            jsonify(
                {
                    "success": True,
                    "product_id": product.product_id,
                    "message": "Product saved successfully",
                }
            ),
            201,
        )

    except Exception as exc:
        session.rollback()
        return (
            jsonify({"success": False, "error": str(exc)}),
            500,
        )
    finally:
        session.close()


@app.route("/api/environment-score", methods=["POST"])
def environment_score():
    """Compute environmental score (cost_score, co2_score, AI score).

    Expected JSON body:
    {
        "cost_values": [1.2, 0.9, 2.0],
        "co2_values": [30.0, 45.0, 20.0]
    }
    or for a single sample:
    {
        "cost_values": 1.2,
        "co2_values": 30.0
    }
    """

    data = request.get_json(silent=True) or {}

    if "cost_values" not in data or "co2_values" not in data:
        return (
            jsonify({"success": False, "error": "cost_values and co2_values are required"}),
            400,
        )

    # Normalize to list
    cost_vals = data["cost_values"]
    co2_vals = data["co2_values"]

    if not isinstance(cost_vals, (list, tuple)):
        cost_vals = [cost_vals]
    if not isinstance(co2_vals, (list, tuple)):
        co2_vals = [co2_vals]

    if len(cost_vals) != len(co2_vals):
        return (
            jsonify({"success": False, "error": "cost_values and co2_values must be same length"}),
            400,
        )

    cost_arr = np.array(cost_vals, dtype=float)
    co2_arr = np.array(co2_vals, dtype=float)

    cost_scores, co2_scores, ai_scores = compute_environment_scores(cost_arr, co2_arr)

    # Convert to Python floats for JSON
    results = []
    for i in range(len(cost_arr)):
        results.append(
            {
                "cost": float(cost_arr[i]),
                "co2": float(co2_arr[i]),
                "cost_score": float(cost_scores[i]),
                "co2_score": float(co2_scores[i]),
                "ai_recommendation_score": float(ai_scores[i]),
            }
        )

    # If single value, return as object instead of list
    if len(results) == 1:
        payload = results[0]
    else:
        payload = results

    return jsonify({"success": True, "results": payload})


@app.route("/api/recommend-materials", methods=["POST"])
def recommend_materials():
    """AI material recommendation endpoint.

    Uses trained models + engineered_dataset.csv to rank materials.

    Expected JSON body:
    {
        "product_weight_kg": 5.0,
        "fragility_level": "Medium",   // Low / Medium / High / Very High
        "top_n": 5
    }
    """

    data = request.get_json(silent=True) or {}

    # Required inputs
    if "product_weight_kg" not in data or "fragility_level" not in data:
        return (
            jsonify({"success": False, "error": "product_weight_kg and fragility_level are required"}),
            400,
        )

    try:
        product_weight = float(data["product_weight_kg"])
    except (TypeError, ValueError):
        return (
            jsonify({"success": False, "error": "product_weight_kg must be numeric"}),
            400,
        )

    fragility = data["fragility_level"]
    top_n = data.get("top_n", 5)

    try:
        result = generate_material_recommendations(product_weight, fragility, top_n)
    except RuntimeError as exc:
        return (
            jsonify({"success": False, "error": str(exc)}),
            500,
        )

    if result["total_candidates"] == 0:
        return (
            jsonify({"success": False, "error": "No materials match product constraints"}),
            404,
        )

    return jsonify({"success": True, **result})


@app.route("/", methods=["GET"])
def index():
    """Simple root endpoint with basic info (for testing)."""

    return jsonify(
        {
            "service": "EcoPack Flask Backend",
            "milestone": 3,
            "module": 5,
            "endpoints": [
                "/api/health",
                "/api/product",
                "/api/environment-score",
                "/api/recommend-materials",
            ],
        }
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Business Intelligence dashboard (Module 7)."""

    if engineered_df is None:
        return render_template(
            "dashboard.html",
            error="Analytics dataset could not be loaded.",
            metrics=None,
            graphs_json=None,
        )

    metrics = compute_dashboard_metrics(engineered_df)
    graphs_json = build_dashboard_figures(engineered_df)

    return render_template(
        "dashboard.html",
        error=None if metrics else "Analytics metrics could not be computed.",
        metrics=metrics,
        graphs_json=graphs_json,
    )


@app.route("/dashboard/export/excel", methods=["GET"])
def export_dashboard_excel():
    """Export sustainability summary as an Excel file."""

    if engineered_df is None:
        return jsonify({"success": False, "error": "Dataset not loaded"}), 500

    metrics = compute_dashboard_metrics(engineered_df)
    if metrics is None:
        return jsonify({"success": False, "error": "Metrics could not be computed"}), 500

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "CO2 Reduction (%)",
                "Cost Savings (%)",
                "Average Recyclability (%)",
                "Average Biodegradability (%)",
            ],
            "Value": [
                metrics["co2_reduction_pct"],
                metrics["cost_savings_pct"],
                metrics.get("avg_recyclability"),
                metrics.get("avg_biodegradability"),
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        # Also include aggregated CO2 and cost by material type
        if {"Material_Type", "CO2_Emission_Score_%", "Cost_per_KG_USD"}.issubset(
            engineered_df.columns
        ):
            agg_df = (
                engineered_df.groupby("Material_Type")[
                    ["CO2_Emission_Score_%", "Cost_per_KG_USD"]
                ]
                .mean()
                .reset_index()
            )
            agg_df.to_excel(writer, index=False, sheet_name="By_Material")

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="ecopack_sustainability_dashboard.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@app.route("/dashboard/export/pdf", methods=["GET"])
def export_dashboard_pdf():
    """Export a simple sustainability report as a PDF file."""

    if engineered_df is None:
        return jsonify({"success": False, "error": "Dataset not loaded"}), 500

    metrics = compute_dashboard_metrics(engineered_df)
    if metrics is None:
        return jsonify({"success": False, "error": "Metrics could not be computed"}), 500

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "EcoPack Sustainability Report")

    y -= 40
    c.setFont("Helvetica", 11)
    lines = [
        f"CO2 reduction (top materials vs overall): {metrics['co2_reduction_pct']:.1f}%",
        f"Cost savings (top materials vs overall): {metrics['cost_savings_pct']:.1f}%",
        f"Average recyclability across materials: {metrics.get('avg_recyclability', 0.0):.1f}%",
        f"Average biodegradability across materials: {metrics.get('avg_biodegradability', 0.0):.1f}%",
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 20

    c.showPage()
    c.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="ecopack_sustainability_report.pdf",
        mimetype="application/pdf",
    )
@app.route("/ui", methods=["GET", "POST"])
def ui_recommendation():
    """HTML UI for material recommendation (Module 6)."""

    error = None
    result = None

    # Default form values
    default_weight = 5.0
    default_fragility = "Medium"
    default_top_n = 5

    if request.method == "POST":
        try:
            product_weight = float(request.form.get("product_weight_kg", default_weight))
        except (TypeError, ValueError):
            error = "Product weight must be numeric."
        else:
            fragility = request.form.get("fragility_level", default_fragility)
            top_n_val = request.form.get("top_n", str(default_top_n))

            try:
                result = generate_material_recommendations(product_weight, fragility, top_n_val)
                if result["total_candidates"] == 0:
                    error = "No materials match the given constraints. Try adjusting inputs."
            except RuntimeError as exc:
                error = str(exc)

    return render_template(
        "materials_ui.html",
        error=error,
        result=result,
        fragility_options=["Low", "Medium", "High", "Very High"],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("=" * 60)
    # print("🌱 EcoPack Flask Backend - Milestone 3 (Module 5)")
    print("Running on http://localhost:%d" % port)
    print("Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/product")
    print("  POST /api/environment-score")
    print("  POST /api/recommend-materials")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=True)
