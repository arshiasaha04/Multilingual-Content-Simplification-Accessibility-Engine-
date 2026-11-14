import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session
from app.database import SimplificationJob
from app.config import Config

class TelemetryAnalyticsGenerator:
    @staticmethod
    def generate_evaluation_charts(db: Session) -> dict:
        records = db.query(SimplificationJob).all()
        if not records:
            return {"status": "empty", "charts": []}

        df = pd.DataFrame([{
            "id": r.id,
            "Language": r.target_language.upper(),
            "Reading_Level": r.reading_level.replace("_", " ").title(),
            "Source_FK_Grade": r.source_flesch_kincaid,
            "Simplified_FK_Grade": r.simplified_flesch_kincaid,
            "Indic_Readability_Index": r.indic_readability_score,
            "BackTranslation_Sim": r.back_translation_cosine_sim,
            "Semantic_Consistency": r.llm_semantic_consistency,
            "Composite_Score": r.composite_quality_score,
            "Iterations": r.iterations_taken
        } for r in records])

        sns.set_theme(style="whitegrid", palette="viridis")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Plot 1: Indic Readability Complexity across Configurable Reading Levels
        sns.barplot(
            data=df,
            x="Reading_Level",
            y="Indic_Readability_Index",
            hue="Language",
            ax=axes[0]
        )
        axes[0].set_title("Indic Readability Complexity by Reading Level (Lower = Simpler)")
        axes[0].set_ylabel("Indic Readability Index (Matra + Conjunct Score)")
        axes[0].set_xlabel("Configured Reading Level")

        # Plot 2: Semantic Consistency vs Back-Translation Similarity
        sns.scatterplot(
            data=df,
            x="BackTranslation_Sim",
            y="Semantic_Consistency",
            hue="Reading_Level",
            style="Language",
            s=120,
            ax=axes[1]
        )
        axes[1].set_title("Quality Validation: Back-Translation vs Semantic Fidelity")
        axes[1].set_xlabel("Back-Translation TF-IDF Cosine Similarity")
        axes[1].set_ylabel("AI Semantic Consistency Score")
        axes[1].set_ylim(0.5, 1.05)

        plt.tight_layout()
        chart_filename = "readability_evaluation_report.png"
        chart_path = os.path.join(Config.PLOTS_OUTPUT_DIR, chart_filename)
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)

        summary_stats = {
            "total_jobs_processed": int(len(df)),
            "avg_fk_grade_reduction": round(float((df["Source_FK_Grade"] - df["Simplified_FK_Grade"]).mean()), 2),
            "avg_indic_readability": round(float(df["Indic_Readability_Index"].mean()), 2),
            "avg_semantic_consistency": round(float(df["Semantic_Consistency"].mean()), 3),
            "avg_self_correction_loops": round(float(df["Iterations"].mean()), 2),
            "chart_url": f"/static/plots/{chart_filename}"
        }
        return summary_stats