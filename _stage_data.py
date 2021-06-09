import sys

from core import Net, DataCluster
from core.tools import safe_div, tic, toc

from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from statistics import mean 
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import logging
import random
import gc
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.signal import find_peaks, savgol_filter

import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from statsmodels.nonparametric.smoothers_lowess import lowess as low


NUM_STOCKS = 0
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform


class Stager:

    def __init__(self, array=None):

        self.array = array
        self.num_time_steps = 60 #number of sequences that will be fed into the model
        self.stage_for = 'val' #train or val

        # Data cluster
        self.dataset = 'realmix'
        self.data_cluster = DataCluster(
            dataset=self.dataset,
            remove_features=['close', 'high', 'low', 'open', 'volume'],
            num_stocks=NUM_STOCKS,
            wavelet_scales=WAVELET_SCALES,
            num_time_steps=self.num_time_steps,
            use_split_data=self.stage_for,
            verbose=True
            )
        self.collection = self.data_cluster.collection

        # Get shape
        # st_shape, lt_shape = self.data_cluster.get_model_shape()
        # print(st_shape, lt_shape), quit()

        # Run
        self.run()


    def run(self):

        # assert requested_target!=None, 'No requested target was speicifed'

        batches = 5_000
        samples_per_batch = 63
        _target_dict = {0:0, 1:0, 2:0}
        
        try:
            for idx, batch in enumerate(range(batches)):

                batch_dict = {'lt':list(), 'st':list(), 'y':list()}

                time_list = list()
                t_0 = time.time()

                # Weights to balance the randomly picked action - resets each batch
                action_weights = {
                    0:int(samples_per_batch/3),
                    1:int(samples_per_batch/3),
                    2:int(samples_per_batch/3)}

                done = False
                while not done:

                    # Random action (target) - 0=Hold, 1=Buy, 2=Sell
                    actions_list = [0]*action_weights[0] + [1]*action_weights[1] + [2]*action_weights[2]
                    random.shuffle(actions_list)
                    requested_action = random.choice(actions_list)
                    requested_action = 1


                    # Sample a data pack from the cluster and setup df
                    self.dp = np.random.choice(self.collection, replace=True)
                    self.df = self.dp.df



                    # Get random start index
                    start_index = random.randint(self.num_time_steps+50,len(self.df)-1)

                    try:
                        self.dp.data_process((start_index,start_index+self.num_time_steps-1))
                    except:
                        continue
                    df_st = self.dp.df_st


                    # FIND PEAKS VALLYES WITH SCIPY
                    _close_df = df_st.close
                    peaks, peak_properties = find_peaks(_close_df, prominence=0.0)
                    valleys, valley_properties = find_peaks(-_close_df, prominence=0.0)
                    peak_prominences = peak_properties['prominences']
                    valley_prominences = valley_properties['prominences']

                    if len(valleys)>0: #doesnt work for empty lists
                        valleys += start_index
                    if len(peaks)>0:
                        peaks += start_index

                    if requested_action == 1:
                        # buy - valleys
                        try:
                            target_index = random.choice(valleys)
                            prominence = valley_prominences[list(valleys).index(target_index)]
                        except Exception as e:
                            print(e)
                            continue

                    elif requested_action == 2:
                        # sell - peaks
                        try:
                            target_index = random.choice(peaks)
                            prominence = peak_prominences[list(peaks).index(target_index)]
                        except:
                            continue

                    elif requested_action == 0:
                        # hold - index that is neither peak nor valley
                        index_not_peak_or_valley = _close_df.index[ (~_close_df.index.isin(peaks)) & (~_close_df.index.isin(valleys)) ].to_list()
                        target_index = random.choice(index_not_peak_or_valley)
                        prominence = 0


                    # aa.plot()
                    # plt.plot(peaks, aa[peaks], "v", c='r')
                    # plt.plot(valleys, aa[valleys], "^", c='g')
                    # plt.show()




                    # Get the df in the span starting from target_index and num_time_steps long
                    requested_span = (target_index - self.num_time_steps + 1, target_index)
                    try:
                        st_features, lt_features = self.dp.data_process(requested_span)
                    except Exception as e:
                        continue

                    
                    # Setup target vector
                    y_target = [0,0,0]
                    # y_target[requested_action] = 1
                    max_prominence = 0.05 #max prominence that are to be consider as 1.0 
                    val = min((prominence / max_prominence)**2, 1)
                    if requested_action == 2:
                        y_target[2] = val
                        y_target[0] = 1 - val
                    elif requested_action == 1:
                        y_target[1] = val
                        y_target[0] = 1 - val

                    # To keep track of hao many targets have been requested
                    _target_dict[requested_action] += 1

                    # Append to dict
                    batch_dict['lt'].append(lt_features)
                    batch_dict['st'].append(st_features)
                    batch_dict['y'].append(y_target)

                    # Balance the action weights
                    action_weights[requested_action] -= 1

                    # Check if done
                    if len(batch_dict['y']) == samples_per_batch:
                        done = True

                t_loop = time.time() - t_0
                time_list.append(t_loop)
                t_avg = sum(time_list) / len(time_list)


                # Pickle batch
                if self.stage_for == 'val':
                    with open(Path.cwd() / 'data' / 'staged' / 'validation' / f'array_{str(self.array)}' / f'staged_batch_{batch}.pkl', 'wb') as handle:
                        pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                elif self.stage_for == 'train':
                    with open(Path.cwd() / 'data' / 'staged' / f'array_{str(self.array)}' / f'staged_batch_{batch}.pkl', 'wb') as handle:
                        pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(f'{self.stage_for} batch {batch} done with len {len(batch_dict["y"])} - Current targets {_target_dict} - Completed in {round(t_avg*(batches-idx)/60,1)} min')

                # Free memory
                del batch_dict
                gc.collect()

        except KeyboardInterrupt:
            print('interrupted!')
            quit()


def split_stock_data(validation_split):


    stock_folder = Path.cwd() / 'data' / 'stock'
    all_files = [x.stem for x in stock_folder.glob('*/')]
    random.shuffle(all_files)

    # Split collection
    split_index = int(len(all_files) * (1-validation_split))
    train_files = all_files[0:split_index]
    val_files = all_files[split_index::]

    # Pickle dump
    with open(Path.cwd() / 'data' / 'staged' / f'train_stocks.pkl', 'wb') as handle:
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path.cwd() / 'data' / 'staged' / f'validation_stocks.pkl', 'wb') as handle:
        pickle.dump(val_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('SPLIT DONE')
    quit()


if __name__ == '__main__':

    ''' USE THIS FUNCTION TO CREATE TEST VAL SPLIT STOCK DATA '''
    # split_stock_data(0.1)

    # Get array number from arg
    _array = sys.argv[1::]
    assert len(_array) != 0, 'Missing array number'
    _array = _array[0]
    print(f'Stager with array {_array}')




    Stager(array=_array)
    # print('=== EOL ===')