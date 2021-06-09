
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import glob
import sys
from pathlib import Path
import os
import inspect
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import DataCluster



NUM_STOCKS = 1
WAVELET_SCALES = 30 #keep - number of frequecys used the wavelet transform

num_time_steps = 30 #number of sequences that will be fed into the model


# Data cluster
dataset = 'realmix'
data_cluster = DataCluster(
    dataset=dataset,
    remove_features=['close', 'high', 'low', 'open', 'volume'],
    num_stocks=NUM_STOCKS,
    wavelet_scales=WAVELET_SCALES,
    num_time_steps=num_time_steps
    )
collection = data_cluster.collection

dp = collection[0]
span = (100, 100+num_time_steps-1)


st, lt = dp.data_process(span=span) 
df = dp.df_st



print(df)
print(lt.shape)

a = np.where( lt > 1 )
print('LT LARGET THAN 1:', len(a))
print(lt[a])

plt.subplot(2, 1, 1)
plt.imshow(lt)

plt.subplot(2, 1, 2)
df.close.plot()

plt.show()
# if len(lt[a]) > 0:
#     plt.show()