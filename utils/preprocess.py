import pandas as pd
from pandas import Series
from sklearn.preprocessing import LabelEncoder

from os import path

# split a dataset into train and test sets
from sklearn.model_selection import train_test_split

from utils.constants import (
    NCBI_PATIENT_METADATA,
    NCBI_NSG_GENE_DATA,
    NCBI_PATIENT_GENE_DATA,
    ANNOTATION_GENE_TO_NAME,
)


def read_data_from_csv():
    if not path.exists(NCBI_PATIENT_GENE_DATA):
        ## file does not exists need create merged file with patinet metadata
        ## and patinet gene expression
        
        ## read 234 patient metadata and Transpose
        ## to match with patient metadata
        ## remove "idseq_sample_name", "SC2_rpm" fields from patient data which is in col 5 and 6
        patient_metadata = pd.read_csv(
            NCBI_PATIENT_METADATA, header=None, low_memory=False #, usecols=[0, 1, 2, 3, 4, 7]
        ).T
        ## read 15,979 gene expression data of 234 patient
        gene_expr_data = pd.read_csv(NCBI_NSG_GENE_DATA, header=None, low_memory=False)
        ## collate patient metdata and gene expression data
        merged = patient_metadata.merge(gene_expr_data, how="outer")
        ## drop last row of the result generated by the merge
        merged.drop(merged.tail(1).index, inplace=True)
        ## Transpose the result so we get genes expression in row
        merged = merged.T
        merged.to_csv(NCBI_PATIENT_GENE_DATA, index=False, header=False)
    return pd.read_csv(NCBI_PATIENT_GENE_DATA, header=0, low_memory=False,)

def load_annotations():
    gene_to_name = pd.read_csv(ANNOTATION_GENE_TO_NAME, header=None, index_col=0, delimiter='\t').to_dict()
    # print(gene_to_name[1])
    return gene_to_name[1]

def get_train_and_test_data():
    ## create dataset
    data = read_data_from_csv()
    data.drop(["CZB_ID", "SC2_rpm", "idseq_sample_name", "sequencing_batch", "viral_status"], axis=1, inplace=True)
    
    # data_columns = data.columns.to_list()
    # # print(data.iloc[0][15979], data.iloc[0][15981])
    # y_columns = ["SC2_PCR"]
    # X_columns = [item for item in data_columns if item not in y_columns]
    # X = data[X_columns]
    # # 15,980 - target column - SC2_PCR
    # y = data[y_columns]
    
    # load gene to name as a dict and change the column values
    gene_to_name = load_annotations()
    data.rename(columns = gene_to_name, inplace = True)
    
    # Use label encode to encode categorical values 
    # https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/?utm_source=blog&utm_medium=Categorical_data_encoding
    number = LabelEncoder()
    print("before encoding")
    print("gender: \n", data["gender"])
    print("SC2: \n",data["SC2_PCR"])
    data["gender"] = number.fit_transform(data["gender"].astype('str'))
    data["SC2_PCR"] = number.fit_transform(data["SC2_PCR"].astype('str'))
    print("after encoding")
    print("gender: \n", data["gender"])
    print("SC2: \n",data["SC2_PCR"])
    X = data.drop("SC2_PCR", axis=1)
    y = data["SC2_PCR"]
    ## split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test
