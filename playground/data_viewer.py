
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import glob
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import DataCluster 


NUM_STOCKS = 1
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform

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
(st_shape, lt_shape) = data_cluster.get_model_shape()
dp = collection[0]

dp.data_process((20,20+num_time_steps-1))