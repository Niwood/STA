import sys

from numpy.core.shape_base import block

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

# import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pandas_ta as ta
from statsmodels.nonparametric.smoothers_lowess import lowess as low

from tslearn.clustering import TimeSeriesKMeans



NUM_STOCKS = 0
WAVELET_SCALES = 30 #keep - number of frequecys used the wavelet transform



'''
1. stage data as usual
2. perform clustering on the staged data to find a peak/valley similarities in the last x timesteps
3. loop through all staged data again to supress buy/sell signals that are not contained in the cluster

'''

class Stager:

    def __init__(self, array=None):

        self.array = array
        self.num_time_steps = 30 #number of sequences that will be fed into the model
        self.stage_for = 'train' #train or val


        # Pre-trained cluster model
        model_name = str(1623307809)
        path = f'cluster_models/{model_name}'        
        self.cluster_model = TimeSeriesKMeans().from_pickle(path+'.pkl')
        self.cluster_model.verbose = False

        with open(path + '_mapping.pkl', 'rb') as handle:
            unique_cluster_list = pickle.load(handle)

        # Create cluster mapping
        self.cluster_mapping = {0:list(), 1:list(), 2:list()}
        for _action in range(3):
            for _cluster in range(30): #num clusters
                try:
                    if unique_cluster_list[_action][_cluster] > 0.75:
                        self.cluster_mapping[_action].append(_cluster)
                except:
                    pass
        
        # print('cluster_mapping: ',self.cluster_mapping)
        # quit()


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
        random.shuffle(self.collection)

        # Get shape
        self.st_shape, self.lt_shape = self.data_cluster.get_model_shape()


        # Run
        self.run()


    def run(self):

        # assert requested_target!=None, 'No requested target was speicifed'

        batches = 3_000
        samples_per_batch = 64
        self._target_dict = {0:0, 1:0, 2:0}
        batch_dict = {'lt':list(), 'st':list(), 'y':list()}
        batch_counter = 0
        batch_index_counter = 0
            
        # time_list = list()
        # t_0 = time.time()

        try:
            for stock_id in range(len(self.collection)):

                # Sample a data pack
                self.dp = self.collection[stock_id]
                df = self.dp.df

                # Process the features and get df_st
                start_index = 30 #some features are constant before this value
                
                self.dp.get_st_features(self.dp.df.copy(), (start_index,len(df)), eng_mode=True)
                df_st = self.dp.df_st
                # print(df_st.columns), quit()

                # Find peaks and valleys with scipy
                _close = df_st.close
                
                peaks_idx, peak_properties = find_peaks(_close, prominence=0.0)
                valleys_idx, valley_properties = find_peaks(-_close, prominence=0.0)
                peaks = pd.DataFrame(data={'df_index':list(peaks_idx), 'prominence':peak_properties['prominences']})
                valleys = pd.DataFrame(data={'df_index':list(valleys_idx), 'prominence':valley_properties['prominences']})

                # Adjust the indicies for the start index
                if len(valleys)>0: #doesnt work for empty lists
                    valleys.df_index += start_index
                if len(peaks)>0:
                    peaks.df_index += start_index

                # Loop though each data point
                counter = 0
                
                for idx in tqdm(df_st.index, desc=f'[{self.dp.ticker}] - {stock_id+1}/{len(self.collection)}'):

                    df_slice = df.loc[idx:idx+self.num_time_steps-1]
                    target_index = df_slice.index[-1]
                    if target_index in peaks.df_index.to_list():
                        character = 'peak'
                        prominence = peaks.loc[peaks.df_index == target_index, 'prominence'].item()
                        
                    elif target_index in valleys.df_index.to_list():
                        character = 'valley'
                        prominence = valleys.loc[valleys.df_index == target_index, 'prominence'].item()
                        
                    else:
                        character = 'none'
                        prominence = 1




                    # Get the df in the span starting from target_index and num_time_steps long
                    requested_span = (target_index - self.num_time_steps + 1, target_index)
                    try:
                        st_features, lt_features = self.dp.data_process(requested_span)
                    except Exception as e:
                        continue


                    # Assert for shape
                    if st_features.shape != self.st_shape:
                        continue
                    elif lt_features.shape != self.lt_shape:
                        continue



                    # Setup target vector
                    y_target = [0,0,0]
                    max_prominence = 1 #max prominence that are to be consider as 1.0 
                    val = min((prominence / max_prominence)**2, 1)


                    if character == 'peak':
                        y_target[2] = val
                        y_target[0] = 1 - val
                    elif character == 'valley':
                        y_target[1] = val
                        y_target[0] = 1 - val
                    elif character == 'none':
                        y_target[0] = 1

                    # print(st_features.shape)
                    # plt.plot(st_features)
                    # plt.title()
                    # plt.savefig("latest_fig.png")
                    # quit()
                    # plt.show()

                    # Run through cluster model
                    # if np.argmax(y_target) in (1,2):
                    #     num_features_to_keep = [3,6,7,8,10,11,12,13,14,15]
                    #     len_for_clustering = 5
                    #     b = st_features[self.num_time_steps-len_for_clustering:,num_features_to_keep]
                    #     b = b.reshape(1,*b.shape)
                    #     y_cluster = self.cluster_model.predict(b)

                    #     if y_cluster not in self.cluster_mapping[np.argmax(y_target)]:
                    #         y_target = [1,0,0]
                    #     else:
                    #         _y_target = [0,0,0]
                    #         _y_target[np.argmax(y_target)] = 1
                    #         y_target = _y_target
                            
                    # else:
                    #     y_target = [1,0,0]
                    
                    


                    # SAMPLE PLOT
                    # if y_target[2] > 0.9:
                    #     _close.loc[requested_span[0]:requested_span[1]+30].plot()
                    #     # self.dp.df_st.close.plot()
                    #     plt.axvline(x=target_index)
                    #     plt.title(f'action {character} {y_target}')
                    #     plt.savefig("latest_fig.png")
                    #     quit()
                    # else:
                    #     continue


                    # Skip sample if the batch has too many hold actions
                    if self.stage_for == 'train':
                        if np.argmax(y_target) == 0 and self._target_dict[0] > max(self._target_dict[1], self._target_dict[2]):
                            continue
                    

                    # -COMMITED TO SAVE THE SAMPLE BEYOND THIS POINT-

                    # To keep track of the targets
                    self._target_dict[0] += y_target[0]
                    self._target_dict[1] += y_target[1]
                    self._target_dict[2] += y_target[2]

                    # Append to batch dict
                    batch_dict['lt'].append(lt_features)
                    batch_dict['st'].append(st_features)
                    batch_dict['y'].append(y_target)


                    # Iterate counters
                    counter += 1

                    # Save batch when reached max samples per batch
                    if len(batch_dict['y']) == samples_per_batch:
                        self.save_batch(batch_index_counter, batch_dict)
                        del batch_dict
                        gc.collect()
                        batch_dict = {'lt':list(), 'st':list(), 'y':list()}
                        batch_counter += 1
                        batch_index_counter += 1
                        if batch_counter % batches == 0:
                            print('EOL --> Stager()')
                            quit()
                            self.array += 1
                            batch_index_counter = 0




        except KeyboardInterrupt:
            print('interrupted!')
            quit()


    def save_batch(self, batch_num, batch_dict):
        # Pickle batch and save
        if self.stage_for == 'val':
            with open(Path.cwd() / 'data' / 'staged' / 'validation' / f'array_{str(self.array)}' / f'staged_batch_{batch_num}.pkl', 'wb') as handle:
                pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif self.stage_for == 'train':
            with open(Path.cwd() / 'data' / 'staged' / f'array_{str(self.array)}' / f'staged_batch_{batch_num}.pkl', 'wb') as handle:
                pickle.dump(batch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        _tot_target = sum([v for k,v in self._target_dict.items()])
        print(f'{self.stage_for} - on array {self.array} - batch {batch_num} done with len {len(batch_dict["y"])} - 0:{round(100*self._target_dict[0]/_tot_target)} 1:{round(100*self._target_dict[1]/_tot_target)} 2:{round(100*self._target_dict[2]/_tot_target)}')



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



class Clustering:
    
    '''
    To find similarities between short sequences (in the end) of some of the technical indicators
    The purpose is to find a comment pattern for buy/sell signals
    '''


    def __init__(self):
        # Load staged data
        self.path_origin = 'data/staged'
        array = 0
        num_batches = 1000
        data = {'st':list(), 'y':list()}
        for _batch in range(num_batches):
            with open(f'{self.path_origin}/array_{array}/staged_batch_{_batch}.pkl', 'rb') as handle:
                _data = pickle.load(handle)
            data['st'] += _data['st']
            data['y'] += _data['y']


        # Parameters
        len_for_clustering = 5
        num_features_to_keep = [3,6,7,8,10,11,12,13,14,15]
        num_time_steps = len(data['st'][0])


        data_block = np.zeros(shape=(len(range(len(data['st']))),len_for_clustering,len(num_features_to_keep)))
        y_true = list()
        for i in range(len(data['st'])):
            a = data['st'][i][num_time_steps-len_for_clustering::]
            data_block[i,:,:] = a[:,num_features_to_keep]
            y_true.append(
                np.argmax(data['y'][i])
                )
        y_true = np.array(y_true)

        # print(data_block.shape)
        # print(data_block)
        

        num_clusters = 30
        cluster_model = TimeSeriesKMeans(
            n_clusters=num_clusters,
            metric="dtw",
            n_init=3,
            n_jobs=3,
            verbose=True)
        y_pred = cluster_model.fit_predict(data_block)

        unique, counts = np.unique(y_pred, return_counts=True)
        all_pred_unique = dict(zip(unique, counts))



        unique_cluster_list = list()
        for _action in range(3):
            unique, counts = np.unique(y_pred[y_true==_action], return_counts=True)
            unique_dict = dict(zip(unique, counts))
            print(unique_dict)
            for k, v in unique_dict.items():
                unique_dict[k] = v/all_pred_unique[k]
            
            unique_cluster_list.append(unique_dict)


        # Save model
        model_name = str(int(time.time()))
        path = f'cluster_models/{model_name}'
        cluster_model.to_pickle(path + '.pkl')

        # Save cluster mapping
        with open(path + '_mapping.pkl', 'wb') as handle:
            pickle.dump(unique_cluster_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Model saved as {model_name}')

        # plt.plot(data_block[0,:,:])
        # plt.title(y_true[0])
        # plt.savefig("latest_fig.png")


if __name__ == '__main__':

    ''' USE THIS FUNCTION TO CREATE TEST VAL SPLIT STOCK DATA '''
    # split_stock_data(0.1)



    ''' USE THIS FOR STAGE DATA '''
    _array = sys.argv[1::]
    assert len(_array) != 0, 'Missing array number'
    _array = _array[0]
    print(f'Stager with array {_array}')
    Stager(array=_array)


    ''' USE THIS FOR CLUSTERING '''
    # Clustering()


    print('=== EOL ===')