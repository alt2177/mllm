import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("drugsComTest_raw.tsv", sep = "\t")
    print(df["review"][0])

if __name__ == "__main__":
    main()
