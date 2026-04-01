import pandas as pd

from app import DATABASE_URL, Product, Base, init_db, SessionLocal
from sqlalchemy import create_engine


def main():
    # Ensure DB engine/session are initialized
    if not init_db():
        raise SystemExit("Failed to connect to database using DATABASE_URL")

    df = pd.read_csv("products.csv")

    required_cols = [
        "Product_Name",
        "Category",
        "Product_Weight_kg",
        "Fragility_Level",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"products.csv is missing columns: {missing}")

    records = []
    for _, row in df.iterrows():
        records.append(
            Product(
                product_name=row["Product_Name"],
                category=row["Category"],
                product_weight_kg=float(row["Product_Weight_kg"]),
                fragility_level=str(row["Fragility_Level"]),
            )
        )

    session = SessionLocal()
    try:
        # Optional: create tables if they do not exist
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)

        session.add_all(records)
        session.commit()
        print(f"Inserted {len(records)} rows into products table.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
