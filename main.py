from core import Net, DataCluster, ModelAssessment, StockTradingEnv

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess as low
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
import pickle
import glob
import time
from tqdm import tqdm
import tensorflow as tf

from tslearn.clustering import TimeSeriesKMeans

import tensorflow.keras.backend as K


# Environment settings
WAVELET_SCALES = 100 #keep - number of frequecys used the wavelet transform

#  Stats settings
EPOCH_SIZE = 100




class Trader:
    '''
    Supervised Trader Algortihm

    '''
    def __init__(self):
        self.st_shape, self.lt_shape = self.get_model_shape()
        # self.lt_shape = (*self.lt_shape, 1)
        
        # self.st_shape = (20, 16)
        # self.lt_shape = (100, 20, 1)

        # print(self.st_shape)
        # print(self.lt_shape)
        # quit()

    def train(self):

        self.model_name = str(int(time.time()))

        # Network
        self.net = Net(self.model_name, (self.st_shape, self.lt_shape))

        # Train
        self.net.train(epochs=50)

        # Save model
        self.folder = Path.cwd() / 'models'
        self.net.model.save(self.folder / self.model_name)

        print(f'Model saved as: {self.model_name}')



    def get_model_shape(self):
        with open('data/staged/array_0/staged_batch_0.pkl', 'rb') as handle:
            _data = pickle.load(handle)
        # print(np.array(_data['st']).shape[1::])
        # print(np.array(_data['lt']).shape[1::])
        # quit()
        return np.array(_data['st']).shape[1::], np.array(_data['lt']).shape[1::]



class Backtest:


    def __init__(self):
        # self.st_shape = (60, 9)
        # self.lt_shape = (100, 60, 1)
        # self.st_shape = (20, 16)
        # self.lt_shape = (100, 20, 1)
        self.st_shape, self.lt_shape = self.get_model_shape()

    def _load_model(self, model_name):
        # Load a model
        model_path = Path.cwd() / 'models' / model_name
        return tf.keras.models.load_model(
            model_path
        )
    
    def get_model_shape(self):
        with open('data/staged/array_0/staged_batch_0.pkl', 'rb') as handle:
            _data = pickle.load(handle)
        # print(np.array(_data['st']).shape[1::])
        # print(np.array(_data['lt']).shape[1::])
        # quit()
        return np.array(_data['st']).shape[1::], np.array(_data['lt']).shape[1::]



    def backtest(self, model_name):
        # Backtest a model

        self.num_time_steps = 30

        # Load model
        model = self._load_model(model_name)

        data_cluster = DataCluster(
            dataset='realmix',
            remove_features=['close', 'high', 'low', 'open', 'volume'],
            num_stocks=100,
            wavelet_scales=30,
            num_time_steps=self.num_time_steps,
            validation_split=0.1
            )
        # train_collection, val_collection = data_cluster.split_collection(0.1)

        # Pre-trained cluster model
        cluster_model_name = str(1623061895)
        path = f'cluster_models/{cluster_model_name}.pkl'        
        self.cluster_model = TimeSeriesKMeans().from_pickle(path)
        self.cluster_model.verbose = False


        collection = data_cluster.collection
        MAX_STEPS = 30*12
        env = StockTradingEnv(collection, 30, max_steps=MAX_STEPS)

        # Eval lists
        stats = {
            'buy': list(),
            'buy_pred': list(),
            'sell': list(),
            'sell_pred': list(),
            'rewards': list(),
            'buy_n_hold': list(),
            'asset_amount': list(),
            
            }
        
        _obs = env.reset()
        obs = {'st':_obs['st'], 'lt':_obs['lt']}
        reward = 0
        ticker = env.ticker

        for _ in tqdm(range(MAX_STEPS), desc=f'Model assessment on {ticker}'):

            # Make prediction

            a = [obs['st'].reshape((1,) + self.st_shape), obs['lt'].reshape((1,) + self.lt_shape)]


            
            prediction = model.predict(a)[0]
            # prediction = [0,0,1]
            action = np.argmax(prediction)

            # Buy/sell threshold
            if action in (1,2):
                if max(prediction) < 0.95:
                    action = 0
            

            # Run through cluster model
            if action in (1,2):
                num_features_to_keep = [3,6,7,8,10,11,12,13,14,15]
                len_for_clustering = 5
                b = obs['st'][self.num_time_steps-len_for_clustering:,num_features_to_keep]
                b = b.reshape(1,*b.shape)
                y_cluster = self.cluster_model.predict(b)

                if y_cluster != action:
                    action = 0


            # Step env
            obs, reward, done = env.step(action)

            if action == 1:
                stats['buy'].append(env.buy_n_hold)
                stats['sell'].append(np.nan)
                stats['buy_pred'].append((max(prediction)+5)**2)
                stats['sell_pred'].append(np.nan)
            elif action == 2:
                stats['buy'].append(np.nan)
                stats['sell'].append(env.buy_n_hold)
                stats['buy_pred'].append(np.nan)
                stats['sell_pred'].append((max(prediction)+5)**2)
            else:
                stats['buy'].append(np.nan)
                stats['sell'].append(np.nan)
                stats['buy_pred'].append(np.nan)
                stats['sell_pred'].append(np.nan)

            stats['rewards'].append(reward)
            stats['asset_amount'].append(env.shares_held)
            stats['buy_n_hold'].append(env.buy_n_hold)

            # Break if done
            if done: break
            
        x = list(range(MAX_STEPS))

        plt.subplot(3,1,1)
        plt.plot(stats['rewards'])

        plt.subplot(3,1,2)
        plt.plot(stats['buy_n_hold'])
        plt.scatter(x, stats['buy'], s=stats['buy_pred'], c="g", alpha=0.5, marker='^',label="Buy")
        plt.scatter(x, stats['sell'], s=stats['sell_pred'], c="r", alpha=0.5, marker='v',label="Sell")

        plt.subplot(3,1,3)
        plt.plot(stats['asset_amount'])
        plt.show()

        print()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    t = Trader()
    t.train()
    
    # b = Backtest()
    # b.backtest('1622759171')

    print('=== EOL: main.py ===')

