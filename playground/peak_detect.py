import os
import inspect
import sys
from matplotlib import pyplot as plt
import random
import numpy as np
from scipy.signal import find_peaks, savgol_filter

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
df = dp.df.close






# Get random start index
start_index = 30

# dp.data_process((start_index,start_index+num_time_steps-1))

dp.get_st_features(dp.df, (start_index,len(df)), eng_mode=True)
df_st = dp.df_st


df_st = df_st.iloc[0:num_time_steps]

# print(df_st)
# quit()

peaks, peak_properties = find_peaks(df_st.close, prominence=0.0)
valleys, valley_properties= find_peaks(-df_st.close, prominence=0.0)
peak_prominences = peak_properties['prominences']
valley_prominences = valley_properties['prominences']


if len(valleys)>0: #doesnt work for empty lists
    valleys += start_index
if len(peaks)>0:
    peaks += start_index



print(f'Num peaks {len(peaks)}')
print(f'Num valleys {len(valleys)}')


def label_distribution(point, prominence):
    max_prominence = 1
    val = min((prominence / max_prominence)**2, 1)
    # if val < 0.3:
    #     val = 0
    out = [0,0,0]
    if point == 'peak':
        out[2] = val
    elif point == 'valley':
        out[1] = val
    # return out
    return val


for idx, prom in enumerate(peak_prominences):
    peak_prominences[idx] = label_distribution('peak', prom)
for idx, prom in enumerate(valley_prominences):
    valley_prominences[idx] = label_distribution('valley', prom)


''' PLOT '''
df_st.close.plot()
# df.plot(subplots=True)

plt.plot(peaks, df[peaks], "v", c='r')
for i, txt in enumerate(peak_prominences):
    plt.annotate(round(txt,3), (peaks[i], df_st.close.loc[peaks].to_list()[i]))

plt.plot(valleys, df[valleys], "^", c='g')
for i, txt in enumerate(valley_prominences):
    plt.annotate(round(txt,3), (valleys[i], df_st.close.loc[valleys].to_list()[i]))

plt.show()
# plt.savefig("latest_fig.png")