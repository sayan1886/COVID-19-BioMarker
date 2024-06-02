import os

__dir_name__ = os.path.dirname(__file__)

DEBUG = True

DATASET_DIR = os.path.join(__dir_name__, "./../dataset/")
NCBI_PATIENT_METADATA = os.path.join(
    __dir_name__, "./../dataset/metatable_with_viral_status.csv"
)
NCBI_NSG_GENE_DATA = os.path.join(__dir_name__, "./../dataset/swab_gene_counts.csv")
NCBI_PATIENT_GENE_DATA = os.path.join(
    __dir_name__, "./../dataset/patient_gene_merged.csv"
)

ANNOTATION_GENE_TO_NAME = os.path.join(
    __dir_name__, "./../dataset/gene2name.txt"
)

LIME_EXPLANATION_HTML = os.path.join(
    __dir_name__, "./../lime/explanation.html"
)