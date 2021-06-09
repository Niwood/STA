# from tensorflow import keras

from tensorflow.keras.utils import Sequence

# from keras.layers import LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.layers import  LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation, Conv1D, GlobalAveragePooling1D, Bidirectional


from tensorflow.keras.optimizers import Adam
# from keras.optimizers.schedules import ExponentialDecay
# from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, MeanAbsoluteError

from tensorflow.keras.models import Model
from tensorflow import Tensor

from numpy.core.numeric import full
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix


from tensorflow.keras.metrics import AUC
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from core.tools import ModifiedTensorBoard
import random
from statsmodels.nonparametric.smoothers_lowess import lowess as low
from tqdm import tqdm
from statistics import mean
import pickle
import math
from pathlib import Path
import glob
from datetime import datetime

from core.tools import safe_div, tic, toc

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# tf.config.run_functions_eagerly(True)

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

class Net:

    def __init__(self, name, model_shape):

        # Constants
        self.model_name = name

        # lt/st shape
        self.st_shape = model_shape[0]
        self.lt_shape = model_shape[1]

        # Create model
        self.model = self._create_model()

        # print(self.model.summary())
        # quit()


        # Save initial model weights
        self.initial_model_weights = np.array(self.model.get_weights()).ravel()
        
        # Parameters
        self.conf_mat = np.array([])
        self.recall = list()
        self.loss = list()
        self.auc = list()
        self.val_loss = list()
        self.val_recall = list()


    def _create_model(self):
        
        dropout_rate = 0.5

        ''' SHORT TERM HEAD '''
        # print(self.st_shape), quit()
        st_head = Input(shape=self.st_shape)


        ### st = Adaptive_Normalizer_Layer(num_features=16)(st_head)
        # st = Dense(self.st_shape[1])(st_head)


        ## Bidirectional LSTM

        # st = Bidirectional(LSTM(10, return_sequences=True))(st_head)
        # st = Bidirectional(LSTM(10, return_sequences=True))(st)
        # st = Bidirectional(LSTM(10, return_sequences=True))(st)
        # st = Bidirectional(LSTM(10, return_sequences=False))(st)

        # st = Flatten(name='stflatten')(st)



        ## ResConvNet
        num_filters = 8
        st = Conv1D(filters=num_filters, kernel_size=3, strides=1 , padding='same', activation='relu')(st_head)


        num_blocks_list = [2, 5, 5, 5, 2]
        
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                st = self.residual_conv1d_block(st, downsample=(j==0 and i!=0), filters=num_filters)
            num_filters *= 2

        st = GlobalAveragePooling1D()(st)


        ## ConvNet
        # st = Conv1D(filters=64, kernel_size=3, padding="same")(st_head)
        # st = BatchNormalization()(st)
        # st = ReLU()(st)

        # st = Conv1D(filters=64, kernel_size=3, padding="same")(st)
        # st = BatchNormalization()(st)
        # st = ReLU()(st)

        # st = Conv1D(filters=64, kernel_size=3, padding="same")(st)
        # st = BatchNormalization()(st)
        # st = ReLU()(st)                        

        # st = Conv1D(filters=64, kernel_size=3, padding="same")(st)
        # st = BatchNormalization()(st)
        # st = ReLU()(st)

        # st = GlobalAveragePooling1D()(st)


        ## ST TAIL

        st = Dense(32)(st)
        st = Dropout(dropout_rate)(st)

        st = Dense(16)(st)
        st = Dropout(dropout_rate)(st)



        ''' LONG TERM HEAD '''
        num_filters = 8
        lt_head = Input(shape=self.lt_shape)

        # lt = BatchNormalization()(lt_head)
        lt = Conv2D(filters=num_filters, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(lt_head)



        # num_blocks_list = [2, 5, 5, 5, 2, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                lt = self.residual_conv2d_block(lt, downsample=(j==0 and i!=0), filters=num_filters)
            num_filters *= 2

        lt = MaxPooling2D(pool_size=4, padding='same')(lt)

        lt = Flatten(name='ltflatten')(lt)

        lt = Dense(32)(lt)
        lt = Dropout(dropout_rate)(lt)

        lt = Dense(16)(lt)
        lt = Dropout(dropout_rate)(lt)

        ''' MERGED TAIL '''
        tail = Concatenate()([st, lt])
        
        tail = Dense(16)(tail)
        tail = Dropout(dropout_rate)(tail)

        tail = Dense(8)(tail)
        tail = Dropout(dropout_rate)(tail)



        ''' OUTPUT '''
        action_prediction = Dense(3, activation='softmax')(tail)
        
        # Compile model
        model = Model(inputs=[st_head, lt_head], outputs=action_prediction)
        opt = Adam(learning_rate=1e-5)
        model.compile(
            loss='categorical_crossentropy',
            optimizer = opt,
            metrics = ['Precision', 'Recall', AUC(curve='PR')]
        )

        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

        self.name = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/fit/" + self.name
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        return model


    def relu_batchnorm(self, inputs: Tensor) -> Tensor:
        # Helper function for relu activation and batch norm
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn


    def residual_conv1d_block(self, x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        # Helper function for a residual block
        y = Conv1D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = self.relu_batchnorm(y)
        y = Conv1D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)

        if downsample:
            x = Conv1D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = self.relu_batchnorm(out)
        return out


    def residual_conv2d_block(self, x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        # Helper function for a residual block
        y = Conv2D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = self.relu_batchnorm(y)
        y = Conv2D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)

        if downsample:
            x = Conv2D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = self.relu_batchnorm(out)
        return out


    def compare_initial_weights(self):
        model_weights = np.array(self.model.get_weights()).ravel()
        b = list()
        for _we1, _we2 in zip(self.initial_model_weights, model_weights):
            a = np.square(np.subtract(_we1,_we2)).mean()
            if a>0: b.append(a) #zero if no bias

        return mean(b)


    def _get_array_size(self):
        num_of_train_arrays = len(glob.glob(f'data/staged/array_*'))
        num_of_val_arrays = len(glob.glob(f'data/staged/validation/array_*'))
        assert num_of_train_arrays > 0 , 'No train arrays found'
        assert num_of_val_arrays > 0 , 'No validation arrays found'
        train_batches_in_array = len(glob.glob(f'data/staged/array_0/staged_batch_*.pkl'))
        val_batches_in_array = len(glob.glob(f'data/staged/validation/array_0/staged_batch_*.pkl'))
        print(num_of_train_arrays, num_of_val_arrays, train_batches_in_array, val_batches_in_array)

        return num_of_train_arrays, num_of_val_arrays, train_batches_in_array, val_batches_in_array


    def train(self, epochs=0):

        # Get array sizes
        num_of_train_arrays, num_of_val_arrays, train_batches_in_array, val_batches_in_array = self._get_array_size()
        # print('TRAIN:', num_of_train_arrays, train_batches_in_array)
        num_of_train_arrays = 3
        num_of_val_arrays = 1
        train_batches_in_array = 2_900
        val_batches_in_array = 2_900

        train_range = list(range(num_of_train_arrays*train_batches_in_array))
        val_range = list(range(num_of_val_arrays*val_batches_in_array))

        # Data generator
        generator_train = DataGenerator(
            train_range,
            num_of_arrays=num_of_train_arrays,
            batches_in_array=train_batches_in_array,
            mode='train'
            )
        generator_val = DataGenerator(
            val_range,
            num_of_arrays=num_of_val_arrays,
            batches_in_array=val_batches_in_array,
            mode='val'
            )

        # Train
        self.model.fit(
            generator_train,
            epochs=epochs,
            validation_data = generator_val,
            callbacks = [self.tensorboard_callback]
            )
        


class DataGenerator(Sequence):
    'Generates data for Keras'


    def __init__(self, list_IDs, shuffle=True, num_of_arrays=0, batches_in_array=0, mode=None):
        
        'Initialization'
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.num_of_arrays = num_of_arrays
        self.batches_in_array = batches_in_array

        if mode == 'train':
            self.path_origin = 'data/staged'
        elif mode == 'val':
            self.path_origin = 'data/staged/validation'
        else:
            print('Missing mode in DataGenerator')
            quit()


    def __len__(self):
        'Number of batches per epoch'
        return int(np.floor(len(self.list_IDs)))


    def __getitem__(self, index):
        'Generate one batch of data'
        bins = np.arange(0, self.num_of_arrays*self.batches_in_array, self.batches_in_array).tolist()
        array = np.digitize(index,bins) - 1
        array += 1

        batch_nums = np.arange(0, self.batches_in_array).tolist() * self.num_of_arrays
        batch_num = batch_nums[index]
        return self.__data_generation(batch_num, array)


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch_num, array):
        with open(f'{self.path_origin}/array_{array}/staged_batch_{batch_num}.pkl', 'rb') as handle:
            _data = pickle.load(handle)
        return [np.array(_data['st']), np.array(_data['lt'])] , np.array(_data['y'])





class Adaptive_Normalizer_Layer(tf.keras.layers.Layer):
    def __init__(self, num_features=2):
        super(Adaptive_Normalizer_Layer, self).__init__()
        
        '''
        source: https://github.com/AntoBr96/Keras_Deep_Adaptive_Input_Normalization/tree/v1.1
        PARAMETERS
        :param mode: Type of normalization to be performed.
            - 'adaptive_average' performs the adaptive average of the inputs
            - 'adaptive_scale' performs the adaptive z-score normalization of the inputs
            - 'full' (Default) performs the complete normalization process: adaptive_average + adaptive_scale + gating
        :param num_features: number of features
        '''

        self.x = None

        self.eps = 1e-8
        
        initializer = tf.keras.initializers.Identity()
        gate_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

        self.linear_1 = tf.keras.layers.Dense(num_features, kernel_initializer=initializer, use_bias=False)
        self.linear_2 = tf.keras.layers.Dense(num_features, kernel_initializer=initializer, use_bias=False)
        self.linear_3 = tf.keras.layers.Dense(num_features, kernel_initializer=gate_initializer)


    def call(self, inputs):
        # Expecting (n_samples, dim, n_feature_vectors) 

        def adaptive_avg(inputs):
            # inputs has the shape: batchsize(64), samples(20 aka num_time_steps), features(16 for st)

            avg = tf.keras.backend.mean(inputs, 1) #changed to 1 axis instead of 2
            adaptive_avg = self.linear_1(avg) #shape: (None (batchsize), input_dim (features)) eg. (64,16)=1024

            # Subtract the adaptive average value for each feature in each batch
            # Reshape input before subtraction as (samples, batchsize, features)
            x = tf.subtract(tf.reshape(inputs,[inputs.shape[1], -1, inputs.shape[2]]), adaptive_avg)

            # Reshape back to (batchsize, samples, features)
            return tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2]])

        
        def adaptive_std(x):
            # x shape: (batchsize, samples, features)
            
            std = tf.keras.backend.mean(x ** 2, 1) #changed axis 2 to 1
            std = tf.keras.backend.sqrt(std + self.eps)

            adaptive_std = self.linear_2(std)
            adaptive_std = tf.where(tf.math.less_equal(adaptive_std, self.eps), 1.0, adaptive_std)

            x = tf.divide(
                tf.reshape(x, [x.shape[1], -1, x.shape[2]]),
                adaptive_std
                )

            # Reshape back to (batchsize, samples, features)
            return tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2]])
        

        def gating(x):
            # x shape: (batchsize, samples, features)

            gate = tf.keras.backend.mean(x, 1)#changed axis 2 to 1
            gate = self.linear_3(gate)
            gate = tf.math.sigmoid(gate)
            
            x = tf.math.multiply(tf.reshape(x,[x.shape[1], -1, x.shape[2]]), gate)

            # Reshape back to (batchsize, samples, features)
            return tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2]])
        

        self.x = adaptive_avg(inputs)
        self.x = adaptive_std(self.x)
        self.x = gating(self.x)
        

        return self.x



if __name__ == '__main__':
    pass
    print('=== EOL ===')

    
