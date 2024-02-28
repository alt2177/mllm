
import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt
from smd.data.load_drug_data import load_drug_data

def main():
    df_training, df_testing = load_drug_data()
    print(df_training.head())


if __name__ == "__main__":
    main()
