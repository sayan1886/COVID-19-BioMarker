import pandas as pd
import os
from pathlib import Path

__dir_name__ = os.path.dirname(Path(__file__).resolve().parent)
dataset_dir = os.path.join(__dir_name__, "dataset")

gene_expr_data = pd.read_csv(os.path.join(dataset_dir, "swab_gene_counts.csv"), header=None, low_memory=False).T
gene_to_name = pd.read_csv(os.path.join(dataset_dir, "gene2name.txt"), header=None, delimiter='\t').T

gene_column = gene_expr_data.iloc[0]
gene_column.drop(index=0, inplace=True, axis=1)
name_column = gene_to_name.iloc[0]
list1 = gene_column.tolist()
list2 = name_column.tolist()
unique_columns = []
gene_name = []
for item in list1:
    if item in list2:
        print(item + " exists ")
        index = list2.index(item)
        name = gene_to_name.iloc[1][index]
        gene_name.append(name)
        print("corresponding gene name ", name)
    else:
        unique_columns.append(item)
print(unique_columns)
print(gene_name)

# gene_expr_data = pd.read_csv(NCBI_NSG_GENE_DATA, header=None, low_memory=False).T
# gene_to_name = pd.read_csv(ANNOTATION_GENE_TO_NAME, header=None, delimiter='\t').T

# gene_column = gene_expr_data.iloc[0]
# gene_column.drop(index=0, inplace=True, axis=1)
# name_column = gene_to_name.iloc[0]
# gene_column_list = gene_column.tolist()
# name_column_list = name_column.tolist()
# unique_columns = []
# gene_name_dict = {}
# for item in gene_column_list:
#     if item in name_column_list:
#         # print(item + " exists ")
#         index = name_column_list.index(item)
#         name = gene_to_name.iloc[1][index]
#         gene_name_dict[item] = name
#         # print("corresponding gene name ", name)
#     else:
#         unique_columns.append(item)
# # print(unique_columns)
# # print(gene_name_dict)
# return gene_name_dict