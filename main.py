import os
import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt

from mllm.data.load_drug_data import load_drug_data
from mllm.core.MLLM import MLLM 
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

def main():
    os.chdir("mergekit")
    df_training, df_testing = load_drug_data()
    print(df_training.head())
    model = MLLM()


if __name__ == "__main__":
    main()
