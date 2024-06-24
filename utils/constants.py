from pathlib import Path
import os

__dir_name__ = os.path.dirname(Path(__file__).resolve().parent)

DEBUG = True

DATASET_DIR = os.path.join(__dir_name__, "dataset")

NCBI_PATIENT_METADATA = os.path.join(
    DATASET_DIR, "metatable_with_viral_status.csv"
)
NCBI_NSG_GENE_DATA = os.path.join(DATASET_DIR, "swab_gene_counts.csv")
NCBI_GENE_COLUMN_DATA = os.path.join(
    DATASET_DIR, "patient_gene_column.csv"
)
NCBI_UNIQUE_GENE_COLUMN_DATA = os.path.join(
    DATASET_DIR, "patient_unique_gene_column.json"
)
NCBI_PATIENT_GENE_DATA = os.path.join(
    DATASET_DIR, "patient_gene_merged.csv"
)

ANNOTATION_GENE_TO_NAME = os.path.join(
    DATASET_DIR, "gene2name.txt"
)

LIME_EXPLANATION_DIR = os.path.join(__dir_name__, "lime")
LIME_EXPLANATION_HTML = os.path.join(
    LIME_EXPLANATION_DIR, "explanation.html"
)

SHAP_EXPLANATION_DIR = os.path.join(__dir_name__, "shap")
SHAP_EXPLANATION_GLOBAL_PLOT = os.path.join(
    SHAP_EXPLANATION_DIR, "shap_global.png"
)
SHAP_EXPLANATION_BAR_PLOT = os.path.join(
    SHAP_EXPLANATION_DIR, "shap_bar.png"
)
SHAP_EXPLANATION_CLUSTER_PLOT = os.path.join(
    SHAP_EXPLANATION_DIR, "shap_cluster.png"
)
SHAP_EXPLANATION_SCATTER_PLOT = os.path.join(
    SHAP_EXPLANATION_DIR, "shap_scatter.png"
)