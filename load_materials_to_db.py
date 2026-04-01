import pandas as pd

from app import DATABASE_URL, Material, Base, init_db, SessionLocal, engineered_df
from sqlalchemy import create_engine


def main():
    # Ensure DB engine/session are initialized
    if not init_db():
        raise SystemExit("Failed to connect to database using DATABASE_URL")

    # Use already-loaded engineered_df from app if available; otherwise read from CSV
    df = engineered_df
    if df is None:
        df = pd.read_csv("engineered_dataset.csv")

    required_cols = [
        "Material_Type",
        "Strength_PSI",
        "Weight_Capacity_KG",
        "Biodegradability_%",
        "CO2_Emission_Score_%",
        "Recyclability_%",
        "Cost_per_KG_USD",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"engineered_dataset.csv is missing columns: {missing}")

    # Map engineered dataset columns to materials table schema
    records = []
    for _, row in df.iterrows():
        records.append(
            Material(
                material_type=row["Material_Type"],
                strength_psi=float(row["Strength_PSI"]),
                weight_capacity_kg=float(row["Weight_Capacity_KG"]),
                biodegradability_percent=float(row["Biodegradability_%"]),
                co2_emission_score_percent=float(row["CO2_Emission_Score_%"]),
                recyclability_percent=float(row["Recyclability_%"]),
                cost_per_kg_usd=float(row["Cost_per_KG_USD"]),
            )
        )

    session = SessionLocal()
    try:
        # Optional: create tables if they do not exist
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)

        session.add_all(records)
        session.commit()
        print(f"Inserted {len(records)} rows into materials table.")
    except Exception as exc:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
