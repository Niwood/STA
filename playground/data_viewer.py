
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import glob
import os
import inspect
import sys
import random
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


from tslearn.clustering import TimeSeriesKMeans

# from core import DataCluster 


# NUM_STOCKS = 1
# WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform

# num_time_steps = 30 #number of sequences that will be fed into the model


# Data cluster
# dataset = 'realmix'
# data_cluster = DataCluster(
#     dataset=dataset,
#     remove_features=['close', 'high', 'low', 'open', 'volume'],
#     num_stocks=NUM_STOCKS,
#     wavelet_scales=WAVELET_SCALES,
#     num_time_steps=num_time_steps
#     )
# collection = data_cluster.collection
# (st_shape, lt_shape) = data_cluster.get_model_shape()
# dp = collection[0]

# dp.data_process((20,20+num_time_steps-1))



array = 0
batch_num = 0

path_origin = Path.cwd().parents[0] / 'data' / 'staged'

with open(f'{path_origin}/array_{array}/staged_batch_{batch_num}.pkl', 'rb') as handle:
    data = pickle.load(handle)


# Pre-trained cluster model
cluster_model_name = str(1623307809)
path = Path.cwd().parents[0] / f'cluster_models/{cluster_model_name}.pkl'
cluster_model = TimeSeriesKMeans().from_pickle(path)
cluster_model.verbose = False


# Get indicies
targets = data['y']
all_hold_indicies = [i for i, x in enumerate(targets) if x == [1,0,0]]
all_buy_indicies = [i for i, x in enumerate(targets) if x == [0,1,0]]
all_sell_indicies = [i for i, x in enumerate(targets) if x == [0,0,1]]



st = data['st']
action = data['y']

# Parameters
features_to_keep = [3,6,7,8,10,11,12,13,14,15]
num_time_steps = data['st'][0].shape[0]
LAST_STEPS = 10
SAMPLES_TO_PLOT = 5


def draw(st, idx, action, plt_idx):
    st = st[idx][num_time_steps-LAST_STEPS:num_time_steps,features_to_keep]

    plt.subplot(SAMPLES_TO_PLOT, 3, plt_idx)
    plt.ylim([0, 1.1])
    if plt_idx < 4:
        plt.title(f'Action {action}')
    plt.plot(st)


for i in range(SAMPLES_TO_PLOT):

    draw(st, random.choice(all_hold_indicies), 'Hold', 3*i+1)
    draw(st, random.choice(all_buy_indicies), 'Buy', 3*i+2)
    draw(st, random.choice(all_sell_indicies), 'Sell', 3*i+3)


# plt.savefig("latest_fig.png")
plt.show()