import os
import matplotlib.pyplot as plt
import polars as pl
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg://biobridge_admin:biobridge_secure_password@localhost:5432/oncosurv_dw"
engine = create_engine(DB_URI)

def run_survival_pipeline():
    print("📥 Adatok betöltése a Bronze táblából...")
    query = "SELECT * FROM bronze.raw_tp53_survival_cohort;"
    df = pl.read_database(query, connection=engine)
    
    # Silver réteg: Adattisztítás és kategorizálás
    print("⚙️ Adatok transzformálása (Silver réteg)...")
    silver_df = df.with_columns([
        pl.when(pl.col("age") < 50).then(pl.lit("<50"))
        .when(pl.col("age") <= 65).then(pl.lit("50-65"))
        .otherwise(pl.lit(">65")).alias("age_group")
    ])
    
    silver_df.write_database(
        table_name="silver.tp53_clinical_cleaned",
        connection=DB_URI,
        if_table_exists="replace"
    )
    print("✅ Silver réteg (silver.tp53_clinical_cleaned) mentve.")

    # Biostatisztikai elemzés: Kaplan-Meier Fitter
    print("📊 Kaplan-Meier illesztés és Log-Rank teszt...")
    pdf = silver_df.to_pandas()
    
    mutants = pdf[pdf["tp53_status"] == "MUTANT"]
    wild_type = pdf[pdf["tp53_status"] == "WILD_TYPE"]

    kmf_mut = KaplanMeierFitter()
    kmf_wt = KaplanMeierFitter()

    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.subplot(111)

    kmf_mut.fit(mutants["survival_months"], event_observed=mutants["vital_status"], label="TP53 Mutant")
    kmf_mut.plot_survival_function(ax=ax, color="#d95f02", ci_show=True)

    kmf_wt.fit(wild_type["survival_months"], event_observed=wild_type["vital_status"], label="TP53 Wild-Type")
    kmf_wt.plot_survival_function(ax=ax, color="#1b9e77", ci_show=True)

    # Log-rank teszt p-érték számítása
    results = logrank_test(
        mutants["survival_months"], wild_type["survival_months"],
        event_observed_A=mutants["vital_status"], event_observed_B=wild_type["vital_status"]
    )
    
    p_value = results.p_value
    print(f"📈 Log-Rank Test p-value: {p_value:.4e}")

    plt.title("Overall Survival: TP53 Mutation vs. Wild-Type (TCGA Cohort)", fontsize=14, fontweight="bold")
    plt.xlabel("Timeline (Months)", fontsize=12)
    plt.ylabel("Overall Survival Probability", fontsize=12)
    plt.annotate(
        f"Log-rank p-value: {p_value:.2e}", 
        xy=(0.05, 0.15), 
        xycoords="axes fraction", 
        fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
    )
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs("docs/assets", exist_ok=True)
    plot_path = "docs/assets/tp53_kaplan_meier.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Túlélési görbe elmentve: {plot_path}")

    # Gold réteg: Kohorsz szintű aggregációk mentése
    print("🏆 Gold metrikák számítása...")
    gold_df = pl.DataFrame({
        "comparison": ["TP53 Mutant vs Wild-Type"],
        "p_value": [float(p_value)],
        "median_survival_mutant": [float(kmf_mut.median_survival_time_)],
        "median_survival_wildtype": [float(kmf_wt.median_survival_time_)],
        "total_cohort_size": [silver_df.height]
    })
    
    gold_df.write_database(
        table_name="gold.survival_biostatistics_summary",
        connection=DB_URI,
        if_table_exists="replace"
    )
    print("✅ Gold réteg (gold.survival_biostatistics_summary) mentve.")

if __name__ == "__main__":
    run_survival_pipeline()