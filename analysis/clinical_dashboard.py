import os
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from lifelines import CoxPHFitter
from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg://biobridge_admin:biobridge_secure_password@localhost:5432/oncosurv_dw"
engine = create_engine(DB_URI)

def generate_clinical_dashboard():
    print("📥 Adatok betöltése a Silver rétegből...")
    query = "SELECT * FROM silver.tp53_clinical_cleaned;"
    df = pl.read_database(query, connection=engine).to_pandas()

    # Stílus és színek beállítása a klinikai publikációs megjelenéshez
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. ÁBRA (Bal fent): Túlélési idő mutációtípusonként (Boxplot)
    ax1 = axes[0, 0]
    sns.boxplot(
        data=df,
        x="mutation_type",
        y="survival_months",
        hue="mutation_type",
        legend=False,
        palette="Set2",
        ax=ax1
    )
    sns.stripplot(
        data=df,
        x="mutation_type",
        y="survival_months",
        color="black",
        alpha=0.3,
        jitter=0.2,
        size=4,
        ax=ax1
    )
    ax1.set_title("A: Overall Survival by TP53 Mutation Class", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Mutation Classification", fontsize=11)
    ax1.set_ylabel("Survival (Months)", fontsize=11)
    ax1.tick_params(axis="x", rotation=15)

    # 2. ÁBRA (Jobb fent): Tumor stádium és vital status arányok
    ax2 = axes[0, 1]
    stage_event = df.groupby(["tumor_stage", "vital_status"]).size().unstack(fill_value=0)
    stage_event_pct = stage_event.div(stage_event.sum(axis=1), axis=0) * 100
    stage_event_pct.plot(
        kind="bar",
        stacked=True,
        color=["#66c2a5", "#fc8d62"],
        ax=ax2,
        edgecolor="none"
    )
    ax2.set_title("B: Mortality Event Proportion across Tumor Stages", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Clinical Tumor Stage", fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.legend(["Alive (Censored)", "Deceased (Event)"], title="Vital Status", loc="upper right")
    ax2.tick_params(axis="x", rotation=0)

    # 3. ÁBRA (Bal lent): Életkor eloszlása TP53 státusz szerint (KDE Plot)
    ax3 = axes[1, 0]
    sns.kdeplot(
        data=df[df["tp53_status"] == "MUTANT"],
        x="age",
        fill=True,
        color="#d95f02",
        label="TP53 Mutant",
        alpha=0.4,
        ax=ax3
    )
    sns.kdeplot(
        data=df[df["tp53_status"] == "WILD_TYPE"],
        x="age",
        fill=True,
        color="#1b9e77",
        label="TP53 Wild-Type",
        alpha=0.4,
        ax=ax3
    )
    ax3.set_title("C: Age Distribution by Genomic Status", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Patient Age at Diagnosis", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.legend(loc="upper right")

    # 4. ÁBRA (Jobb lent): Cox Proportional Hazards (Hazard Ratio Forest Plot)
    ax4 = axes[1, 1]
    print("📈 Cox Proportional Hazards modell számítása...")
    # Modell előkészítése numerikus kódolással
    cox_df = df[["survival_months", "vital_status", "age"]].copy()
    cox_df["is_mutant"] = (df["tp53_status"] == "MUTANT").astype(int)
    cox_df["is_late_stage"] = df["tumor_stage"].isin(["Stage III", "Stage IV"]).astype(int)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="survival_months", event_col="vital_status")
    
    # Forest plot Lifelines beépített plotjával
    cph.plot(ax=ax4)
    ax4.set_title("D: Multivariable Hazard Ratios (Cox Proportional Hazards)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("log(Hazard Ratio) (95% CI)", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Mentés
    os.makedirs("docs/assets", exist_ok=True)
    out_path = "docs/assets/clinical_biostats_dashboard.png"
    plt.suptitle("TCGA-OV Translational Biostatistics & Clinical Cohort Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Dashboard ábra elmentve ide: {out_path}")

if __name__ == "__main__":
    generate_clinical_dashboard()