from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
import pywt
import cv2
import pickle
from scipy.stats import zscore

class DataCluster:
    '''
    Contains a collection of data packs
    '''

    def __init__(
        self,
        dataset=None,
        remove_features=False,
        num_stocks=1,
        wavelet_scales=0,
        num_time_steps=0,
        validation_split=0,
        verbose=False,
        use_split_data=None):

        
        self.dataset = dataset
        self.remove_features = remove_features
        self.num_stocks = num_stocks
        self.wavelet_scales = wavelet_scales
        self.num_time_steps = num_time_steps
        self.validation_split = validation_split
        self.verbose = verbose
        self.use_split_data = use_split_data

        # Compile collection
        self.collection = list()
        self.compile()


    def compile(self):

        # Specify the dir - to get the data folder when a script calls for this package that is in playground folder
        if self.use_split_data == None:

            if 'playground' in Path.cwd().parts:
                stock_folder = Path.cwd().parents[0] / 'data' / 'stock'
            else:
                stock_folder = Path.cwd() / 'data' / 'stock'

            all_files = [x.stem for x in stock_folder.glob('*/')]

        elif self.use_split_data == 'train':
            with open(Path.cwd() / 'data' / 'staged' / f'train_stocks.pkl', 'rb') as handle:
                all_files = pickle.load(handle)

        elif self.use_split_data == 'val':
            with open(Path.cwd() / 'data' / 'staged' / f'validation_stocks.pkl', 'rb') as handle:
                all_files = pickle.load(handle)
        
        # Shuffle and save num of available stocks
        random.shuffle(all_files)
        self.num_stocks_available = len(all_files)


        # Sample
        iterator = range(len(all_files)) if self.num_stocks==0 else range(self.num_stocks)
        files_range = list(range(len(all_files)))

        
        for i in tqdm(
            iterator, desc=f'Generating data cluster'
            ) if self.verbose else iterator:

            skip = False
            while True:
                try:
                    idx = i if self.num_stocks==0 else random.choice(files_range) #Sample an index
                    files_range.remove(idx) #Remove that index from the list
                except:
                    skip = True
                    break
                _file = all_files[idx] #Get the file

                # Resample when the file is empty or unreadable
                try:
                    if 'playground' in Path.cwd().parts:
                        df = pd.read_csv(Path.cwd().parents[0] / 'data' / 'stock' / f'{_file}.txt', delimiter = ",")
                    else:
                        df = pd.read_csv(f'data/stock/{_file}.txt', delimiter = ",")

                except:
                    continue

                # Resample for small dataframes
                if len(df) < 600:
                    continue 

                # All ok -> break
                break

            
            if skip: continue

            # df.set_index('Date', inplace=True)
            # df.drop(['OpenInt'], axis=1, inplace=True)
            # df.index = pd.to_datetime(df.index)
                
            self.collection.append(
                DataPack(
                    dataframe=df,
                    ticker=_file,
                    remove_features=self.remove_features,
                    num_time_steps=self.num_time_steps,
                    wavelet_scales=self.wavelet_scales
                    )
                )

        # Number of features
        # self.num_lt_features = self.collection[0].num_lt_features
        # self.num_st_features = self.collection[0].num_st_features



    def get_model_shape(self):
        ''' Returns the shape that will go into model '''

        # Sample a datapack
        dp = self.collection[0]

        # Arbitrary span for calculation
        span = (300,self.num_time_steps+299)
        
        # Request data process from datapack
        df_st, df_lt = dp.data_process(span)

        return df_st.shape, df_lt.shape





class DataPack:
    '''
    Contains the data for one time serie
    Performes a process of the data incl scaling and feature extraction
    '''

    def __init__(self, dataframe=None, ticker=None, remove_features=False, num_time_steps=0, wavelet_scales=0):

        # Parameters
        self.remove_features = remove_features
        self.ticker = ticker
        self.wavelet_scales = wavelet_scales
        self.num_time_steps = num_time_steps
        self.st_scale_percent = 1 #fraction of original size
        self.lt_scale_percent = 1

        # Load data
        self.df = dataframe

        # Pre-process the data frame
        self.df_process()

        # Add original values to df
        self.org = self.df[['close', 'high', 'low', 'open', 'volume']].copy()

        # Save index as date
        self.date_index = self.df.index.copy()

        # Switch to numeric index
        self.df.index = list(range(len(self.df)))

        # Count features
        # self.count_features()


    def df_process(self):
        
        # Drop col
        self.df.drop(['OpenInt'], axis=1, inplace=True)

        # Set index to datetime
        self.df.set_index('Date', inplace=True)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Rename columns
        self.df.columns= self.df.columns.str.lower()

        # Forward fill for missing dates
        # self.df = self.df.asfreq(freq='1d', method='ffill')



    def data_process(self, span):
        '''
        Process data for feature extraction
        Output as numpy array
        '''

        # assert span[0]>0 , 'Negative span'


        # Slice the df according to the span
        # -50 due to nan values when calculating TIs
        _df = self.df.loc[span[0] - 50 : span[1]].copy()

        # Zscore and scale
        _df = _df.apply(zscore)
        norm_scaler = MinMaxScaler()
        _df[_df.columns] = norm_scaler.fit_transform(_df[_df.columns])

        # Perform feature eng
        st_array = self.get_st_features(_df, span)
        lt_array = self.get_lt_features(_df, span)

        return st_array, lt_array



    def get_st_features(self, df, span, eng_mode=False):
        '''
        SHORT TERM
        '''


        # High/low spread (1)
        df['spread'] = (df.high - df.low).abs()

        

        # Daily return
        # a = df.close.pct_change()
        # a.fillna(0, inplace=True)
        # get all returns as array
        daily_returns = df.close.pct_change().fillna(0).to_numpy()
        

        # sample daily returns with replacement and calculate the mean
        bootstraps = 100
        daily_return_histogram = np.array([])
        for _ in range(bootstraps):
            _sample = np.random.choice(daily_returns, size=self.num_time_steps, replace=True)
            daily_return_histogram = np.append(daily_return_histogram, _sample)
        daily_return_histogram = np.where(daily_return_histogram == np.inf, 0, daily_return_histogram)


        _median = np.median(daily_return_histogram)
        _pct_change = df.close.pct_change().fillna(0).replace([np.inf, -np.inf], 1)
        df['deviation_from_median_return'] = abs(_pct_change - _median)
        df['deviation_from_median_return'][df['deviation_from_median_return'] >= 1] = 1

        

        # MACD (2)
        macd = df.ta.macd(fast=12, slow=26).MACDh_12_26_9
        df['MACD_pos'] = abs(macd) * (macd>0)
        df['MACD_neg'] = abs(macd) * (macd<0)

        
        # RSI signal (3)
        rsi = df.ta.rsi() / 100
        df['RSI'] = rsi.round(4)
        rsi_high = 0.6
        rsi_low = 0.4
        df['RSI_high'] = (rsi * (rsi>rsi_high)).round(4)
        df['RSI_low'] = rsi * (rsi<rsi_low)

        # TRIX signal (1)
        # df['TRIX'] = df.ta.trix(length=14).TRIXs_14_9

        # Bollinger band
        length = 30
        bband = df.ta.bbands(length=length)
        bband['hlc'] = df.ta.hlc3()
        
        

        
        # Bollinger band upper signal - percentage of how close the hlc is to upper bband
        bbu_signal = (bband['hlc']-bband['BBM_'+str(length)+'_2.0'])/(bband['BBU_'+str(length)+'_2.0'] - bband['BBM_'+str(length)+'_2.0'])
        _bbu_no_cap = bbu_signal.loc[span[0] : span[1]].to_list() #used for df_st -> stage_data
        bbu_signal_beyond = (bbu_signal > 1) * (bbu_signal-1)
        bbu_signal_beyond[bbu_signal_beyond > 1] = 1
        bbu_signal[bbu_signal > 1] = 1
        bbu_signal[bbu_signal < 0] = 0
        bband['BBU_signal'] = bbu_signal


        # Bollinger band lower signal
        bbl_signal = (bband['hlc']-bband['BBM_'+str(length)+'_2.0'])/(bband['BBL_'+str(length)+'_2.0'] - bband['BBM_'+str(length)+'_2.0'])
        _bbl_no_cap = bbl_signal.loc[span[0] : span[1]].to_list() #used for df_st -> stage_data

        bbl_signal_beyond = (bbl_signal > 1) * (bbl_signal-1)
        bbl_signal_beyond[bbl_signal_beyond > 1] = 1
        bbl_signal[bbl_signal > 1] = 1
        bbl_signal[bbl_signal < 0] = 0
        bband['BBL_signal'] = bbl_signal

        
        # Append BB signal (4)
        df['BBU_signal'] = bband.BBU_signal
        df['BBU_beyond'] = bbu_signal_beyond
        df['BBL_signal'] = bband.BBL_signal
        df['BBL_beyond'] = bbl_signal_beyond


        # Slice again to remove nan values
        df = df.loc[span[0] : span[1]]

        

        # Drop features that are not supposed to go into model - save close for self.df_st
        # _close = df.close.to_list()
        # df = df.drop(self.remove_features, axis=1)

        # Check null
        df.fillna(method='bfill', inplace=True)
        if df.isnull().sum().sum() > 0:
            print(self.ticker)
            print(f'{self.ticker} - FOUND NULL IN ST FEATURES - see data_loader -> get_st_features - Ticker: {self.ticker}')
            assert False

        

        # Round all values
        df = df.round(4)

        # Save st features as a df - used for staging   
        self.df_st = df.copy()
        # self.df_st['close'] = _close

        
        if not eng_mode:
            # Check so all values in df are between 0-1
            if ((df < 0).any()).any():
                print(df)
                print('THISHDF',(df < 0).any())
                print(f'All ST values not positive - see data loader -> get_st_features - Ticker: {self.ticker}')

                self.df_st.plot(subplots=True)
                plt.savefig("latest_fig.png")
                quit()

            elif ((df > 1).any()).any():
                print(df)
                print(df.describe())
                print(f'All ST values not less than 1 - see data loader -> get_st_features - Ticker: {self.ticker}')
                self.df.plot(subplots=True)
                print(df.max())
                plt.savefig("latest_fig.png")
                quit()

        # Convert to numpy 
        out = df.abs().to_numpy()

        # Resize to increase fit/train speed
        # height = int(out.shape[0] * self.st_scale_percent)
        # out = cv2.resize(out, (out.shape[1], height), interpolation=cv2.INTER_AREA)

        return out


    # def get_st_values_at_index(self, index):
    #     # To get ST values at some index - used as req for staging data
    #     df = self.df_st.loc[index-60:index]
    #     aa = self.get_slice((index-59,index)).close.to_list()
    #     df['close'] = aa

    #     df.plot(subplots=True)
    #     plt.savefig("latest_fig.png")
    #     return self.df_st.loc[index]

        
    def get_lt_features(self, df, span):
        '''
        LONG TERM
        long term features requires to have "LT_" in the beginning of the name
        '''

        # Wavelets as LT
        df['LT_close'] = df.close.copy()

        # Slice again to remove nan values
        df = df.loc[span[0] : span[1]]
        df = df[['LT_close']].abs()

        # Wavelet transform
        wt_trans = self.make_wavelet(df)

        
        # # SHOW WAVELET TRANSFORM
        # plt.imshow(wt_trans, interpolation='nearest')
        # plt.show()
        # quit()

        return wt_trans



    def make_wavelet(self, signals):
        # Outputs wavelet transform, input pandas df
        # coef: (scales, time_steps)

        # Freq scales used in the transform
        scales = np.arange(1, self.wavelet_scales+1)

        # Allocate output
        out = np.zeros(shape=(len(scales), self.num_time_steps, len(signals.columns)))
        
        for idx, col in enumerate(signals.columns):
            signal = signals[col].diff().fillna(0).to_numpy()
            coef, _ = pywt.cwt(signal, scales, wavelet='gaus8') #gaus8 seems to give high res
            
            try:
                out[:, 0:coef.shape[1], idx] = abs(coef)
            except Exception as e:
                print(e)
                print('->>', coef.shape, idx)
                print('->>', out.shape)
                quit()

        # Scale all values
        scaler = MinMaxScaler()
        out = scaler.fit_transform( out.reshape(-1, out.shape[-1]) ).reshape(out.shape)

        # Resize to increase fit/train speed
        # width = int(out.shape[1] * self.lt_scale_percent)
        # height = int(out.shape[0] * self.lt_scale_percent)
        # out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)

        return out #Format: (scales, time_steps, num_LT_features)



    def get_slice(self, span, scale=False):
        ''' To get the close values for a certain span '''
        if not scale:
            return self.df.loc[span[0] : span[1]].copy()
        else:
            _df = self.df.loc[span[0] : span[1]].copy()
            for col in _df.columns:
                _df[col] = _df[col] / _df[col].iloc[0]
            return _df


    def count_features(self):
        ''' Return number of features for long/short term df '''

        _span = (200,300) #Arbitrary span to be able to perform data process
        df_st, df_lt = self.data_process(_span)

        self.num_lt_features = len(df_lt.columns)
        self.num_st_features = len(df_st.columns)





if __name__ == '__main__':
    from core import StockTradingEnv
    
    num_steps = 300
    wavelet_scales = 100
    dc = DataCluster(
        dataset='realmix',
        remove_features=['close', 'high', 'low', 'open', 'volume'],
        num_stocks=5,
        wavelet_scales=wavelet_scales,
        num_time_steps=num_steps
        )

    (st_shape, lt_shape) = dc.get_model_shape()
    print(st_shape)
    quit()
    collection = dc.collection

    env = StockTradingEnv(
        collection,
        look_back_window=num_steps,
        generate_est_targets=True
        )
    env.requested_target = 1
    obs = env.reset()
    


    # df = df[0:300]

    # df.plot(subplots=True)
    # plt.show()


    # def make_data():
    #     dc = DataCluster(
    #         dataset='realmix',
    #         remove_features=['close', 'high', 'low', 'open', 'volume'],
    #         num_stocks=0,
    #         num_time_steps=300
    #         )
    #     collection = dc.collection