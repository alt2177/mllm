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

# |%%--%%| <zaXx0yw20g|3WzLxPYYUB>

# get to correct directory
# os.chdir("./test_loss_exp/")

print(os.getcwd())

#|%%--%%| <3WzLxPYYUB|Fh6sWFkqI7>

# load all probabilities

# ties merge
ties_test = pl.read_csv("test_loss_exp/test_probabilities.csv")
ties_val = pl.read_csv("test_loss_exp/validation_probabilities.csv")

# dare linear merge
dare_lin_test = pl.read_csv("test_loss_exp/dare_linear_test_probabilities.csv")
dare_lin_val = pl.read_csv("test_loss_exp/dare_linear_validation_probabilities.csv")

# GPT2-XL 
gpt2_xl_test = pl.read_csv("test_loss_exp/gpt2_xl_test_probabilities.csv")
gpt2_xl_val = pl.read_csv("test_loss_exp/gpt2_xl_validation_probabilities.csv")


#|%%--%%| <Fh6sWFkqI7|eLq6t46PoP>

# create dataframes with only the probabilities
ties_test_probs = ties_test.select(pl.exclude("true_label"))
ties_val_probs = ties_val.select(pl.exclude("true_label"))

dare_lin_test_probs = dare_lin_test.select(pl.exclude("true_label"))
dare_lin_val_probs = dare_lin_val.select(pl.exclude("true_label"))

gpt2_xl_test_probs = gpt2_xl_test.select(pl.exclude("true_label"))
gpt2_xl_val_probs = gpt2_xl_val.select(pl.exclude("true_label"))

#|%%--%%| <eLq6t46PoP|jphG4SNWO1>

# get distribution of max values
ties_test_maxes = ties_test_probs.with_columns(max = pl.max_horizontal(ties_test_probs.columns))
ties_val_maxes = ties_val_probs.with_columns(max = pl.max_horizontal(ties_val_probs.columns))


dare_lin_test_maxes = dare_lin_test_probs.with_columns(max = pl.max_horizontal(dare_lin_test_probs.columns))
dare_lin_val_maxes = dare_lin_val_probs.with_columns(max = pl.max_horizontal(dare_lin_val_probs.columns))

gpt2_xl_test_maxes = gpt2_xl_test_probs.with_columns(max = pl.max_horizontal(gpt2_xl_test_probs.columns))
gpt2_xl_val_maxes = gpt2_xl_val_probs.with_columns(max = pl.max_horizontal(gpt2_xl_val_probs.columns))


#|%%--%%| <jphG4SNWO1|vvCOfdMBjs>

# Plotting the distribution
plt.figure(figsize = (10, 6), dpi = 100)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')
plt.hist(dare_lin_test_maxes.select("max"), bins = 20, density = True, alpha = 0.3, label = "DARE Linear merge", color='springgreen')
plt.hist(gpt2_xl_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "GPT2-XL", color='violet')
#plt.hist(ties_val_maxes.select("max"), bins = 30, alpha = 0.75, label = "gpt2-ties-merge (validation)", color='violet')
plt.title('Distribution of Maximum Values per Row (Testing)')
plt.xlabel('Max Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig("notebooks/images/max_distributions.png")
plt.show()

#|%%--%%| <vvCOfdMBjs|SbuiSHsCrv>

counts, bins, _ = plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')
counts = (ties_test_maxes.select("max") / sum(ties_test_maxes.select("max"))) * 100

weights = np.ones_like(ties_test_maxes.select("max")) / len(ties_test_maxes.select("max")) * 100
ties_test_maxes.shape
print(weights.shape)
plt.hist(, bins = bins, weights = weights, alpha = 0.5)
# plt.hist(ties_test_maxes.select("max"), bins = 20, density = True, alpha = 0.5, label = "TIES merge", color='dodgerblue')


#|%%--%%| <SbuiSHsCrv|cZWMd5q3yC>









