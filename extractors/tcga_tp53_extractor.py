import numpy as np
import polars as pl
from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg://biobridge_admin:biobridge_secure_password@localhost:5432/oncosurv_dw"
engine = create_engine(DB_URI)

def generate_tp53_cohort(n_patients: int = 450) -> pl.DataFrame:
    np.random.seed(42)

    patient_ids = [f"TCGA-OV-{i:04d}" for i in range(1, n_patients + 1)]
    ages = np.random.normal(loc=58, scale=9.5, size=n_patients).clip(30, 85).astype(int)
    stages = np.random.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], size=n_patients, p=[0.10, 0.15, 0.55, 0.20])

    tp53_status_pool = np.random.choice(["MUTANT", "WILD_TYPE"], size=n_patients, p=[0.68, 0.32])

    mutation_types = []
    survival_times = []
    events = []

    for status in tp53_status_pool:
        if status == "MUTANT":
            m_type = np.random.choice(["Missense", "Nonsense", "Frameshift_Del", "Splice_Site"], p=[0.60, 0.20, 0.15, 0.05])
            surv = np.random.exponential(scale=24.0) + 3.0
            event = np.random.choice([1, 0], p=[0.75, 0.25])
        else:
            m_type = "None (Intact TP53)"
            surv = np.random.exponential(scale=48.0) + 6.0
            event = np.random.choice([1, 0], p=[0.45, 0.55])

        mutation_types.append(m_type)
        survival_times.append(round(float(surv), 2))
        events.append(int(event))

    return pl.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "tumor_stage": stages,
        "tp53_status": tp53_status_pool,
        "mutation_type": mutation_types,
        "survival_months": survival_times,
        "vital_status": events
    })

def load_bronze_tp53():
    print("⏳ Generáljuk a klinikai onkogenomikai kohorszot...")
    df = generate_tp53_cohort()

    print(f"📦 Rekordok száma: {df.height} páciens.")
    print("🚀 Betöltés a PostgreSQL Bronze rétegbe: bronze.raw_tp53_survival_cohort...")

    df.to_pandas().to_sql(
        name="raw_tp53_survival_cohort",
        con=engine,
        schema="bronze",
        if_exists="replace",
        index=False
    )
    print("✅ Betöltés sikeres!")

if __name__ == "__main__":
    load_bronze_tp53()