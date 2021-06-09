
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator
from pathlib import Path
import keras
import numpy as np
import pandas as pd
import pickle
import glob
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, shuffle=True, num_of_arrays=0, batches_in_array=0):
        self._get_size()
        'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.num_of_arrays = num_of_arrays
        self.batches_in_array = batches_in_array

    def _get_size(self):
        self.num_of_arrays = len(glob.glob(f'/home/robin/Documents/STA/data/staged/array_*'))
        assert self.num_of_arrays > 0 , 'No arrays found'
        self.batches_in_array = len(glob.glob(f'/home/robin/Documents/STA/data/staged/array_0/staged_batch_*.pkl'))

    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.list_IDs)))


    def __getitem__(self, index):
        'Generate one batch of data'
        bins = np.arange(0, self.num_of_arrays*self.batches_in_array, self.batches_in_array).tolist()
        array = np.digitize(index,bins) - 1

        batch_nums = np.arange(0, self.batches_in_array).tolist() * self.num_of_arrays
        batch_num = batch_nums[index]
        return self.__data_generation(batch_num, array)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch_num, array):

        with open(f'/home/robin/Documents/STA/data/staged/array_{array}/staged_batch_{batch_num}.pkl', 'rb') as handle:
            _data = pickle.load(handle)
        
        return [np.array(_data['st']), np.array(_data['lt'])] , np.array(_data['y'])


def get_array_size():
    num_of_arrays = len(glob.glob(f'/home/robin/Documents/STA/data/staged/array_*'))
    assert num_of_arrays > 0 , 'No arrays found'
    batches_in_array = len(glob.glob(f'/home/robin/Documents/STA/data/staged/array_0/staged_batch_*.pkl'))
    return num_of_arrays, batches_in_array

num_of_arrays, batches_in_array = get_array_size()
print(num_of_arrays, batches_in_array)

partition = {'train': list(range(num_of_arrays*batches_in_array)), 'validation': [0]}
generator = DataGenerator(
    partition['train'],
    num_of_arrays=num_of_arrays,
    batches_in_array=batches_in_array)

# number of samples
print('Samples: %d' % len(generator))

# print each sample
print('='*5)
for i in range(len(generator)):
    [st, lt], y = generator[i]
    # print(f' => {y} {i}')
    if i == 0:
        lt_shape = lt.shape
        st_shape = st.shape
    if st_shape != st.shape or lt_shape != lt.shape:
        print('i', i)
        print(st_shape, st.shape)
        print(lt_shape, lt.shape)

    # quit()