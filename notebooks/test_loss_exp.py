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
# os.chdir("./test_loss_exp/")

print(os.getcwd())


#|%%--%%| <1vPiao2xcN|KCkw4KfUIQ>

df_test = pl.read_csv("test_loss_exp/test_probabilities.csv")
df_val = pl.read_csv("test_loss_exp/validation_probabilities.csv")

# create dataframes with only the probabilities
df_test_probs = df_test.select(pl.exclude("true_label"))
df_val_probs = df_val.select(pl.exclude("true_label"))

#|%%--%%| <KCkw4KfUIQ|enSpeRK2EP>

df_test.shape
df_test.head()

#|%%--%%| <enSpeRK2EP|jphG4SNWO1>

# get distribution of max values

df_test_maxes = df_test_probs.with_columns(max = pl.max_horizontal(df_test_probs.columns))
df_val_maxes = df_val_probs.with_columns(max = pl.max_horizontal(df_val_probs.columns))

#|%%--%%| <jphG4SNWO1|vvCOfdMBjs>


# Plotting the distribution
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(df_test_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (test)", color='lightskyblue')
plt.hist(df_val_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (validation)", color='violet')
plt.title('Distribution of Maximum Values per Row')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/max_distributions.png")
plt.show()

#|%%--%%| <vvCOfdMBjs|SbuiSHsCrv>




