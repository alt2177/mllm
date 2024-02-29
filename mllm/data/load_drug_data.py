import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas

fs = pyxet.XetFS()

def load_drug_data() -> None:
    df_drugs_train = pl.read_csv('xet://alt2177/mllm-data/main/data/drug_data/drugsComTrain_raw.tsv', separator = '\t')
    df_drugs_test = pl.read_csv('xet://alt2177/mllm-data/main/data/drug_data/drugsComTest_raw.tsv', separator = '\t')

    return df_drugs_train, df_drugs_test
