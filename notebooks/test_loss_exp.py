"""°°°
# Investigating High Test Loss with Drug Dataset 

°°°"""
# |%%--%%| <IeS9KzIxpW|Gz5EGgfL1C>

%pip install pyxet
%pip install polars
%pip install pandas
%pip install numpy
%pip install pyarrow

# |%%--%%| <Gz5EGgfL1C|zaXx0yw20g>

import pyxet         # make xet:// protocol available   
import pandas as pd 
import polars as pl  # faster alternative to pandas
import numpy as np
import pyarrow
import matplotlib.pyplot as plt
import os

fs = pyxet.XetFS()

# |%%--%%| <zaXx0yw20g|1vPiao2xcN>

# get to correct directory
os.chdir("../test_loss_exp")

print(os.getcwd())


#|%%--%%| <1vPiao2xcN|KCkw4KfUIQ>

df_test = pl.read_csv("test_probabilities.csv")
df_val = pl.read_csv("validation_probabilities.csv")

#|%%--%%| <KCkw4KfUIQ|enSpeRK2EP>

df_test.shape

#|%%--%%| <enSpeRK2EP|jphG4SNWO1>

df_test.head()

#|%%--%%| <jphG4SNWO1|SbuiSHsCrv>



