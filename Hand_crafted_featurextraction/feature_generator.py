import pandas as pd
import scipy
import numpy as np
import os
import scipy.stats
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

import copy
import scipy.signal


#ref : https://github.com/fraunhoferportugal/tsfel/blob/master/tsfel/f
def kurtosis(signal):
    """Computes kurtosis of the signal.
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which kurtosis is computed
    Returns
    -------
    float
        Kurtosis result
    """
    return scipy.stats.kurtosis(signal)

def skewness(signal):
    """Computes skewness of the signal.
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which skewness is computed
    Returns
    -------
    int
        Skewness result
    """
    return scipy.stats.skew(signal)


def entropy(signal, prob='standard'):
        """Computes the entropy of the signal using the Shannon Entropy.
        Description in Article:
        Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
        Authors: Crutchfield J. Feldman David
        
        Parameters
        ----------
        signal : nd-array
        Input from which entropy is computed
        prob : string
        Probability function (kde or gaussian functions are available)
        Returns
        -------
        float
        The normalized entropy value
        """

        if prob == 'standard':
            value, counts = np.unique(signal, return_counts=True)
            p = counts / counts.sum()
        elif prob == 'kde':
            p = kde(signal)
        elif prob == 'gauss':
            p = gaussian(signal)

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        # If probability all in one value, there is no entropy
        if np.log2(len(signal)) == 1:
            return 0.0
        elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
            return 0.0
        else:
            return - np.sum(p * np.log2(p)) / np.log2(len(signal))

def cal_max(signal):
    """Computes the maximum value of the signal.
    Parameters
    ----------
    signal : nd-array
       Input from which max is computed
    Returns
    -------
    float
        Maximum result
    """
    return np.max(signal)


def cal_min(signal):
    """Computes the minimum value of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which min is computed
    Returns
    -------
    float
        Minimum result
    """
    return np.min(signal)



def cal_mean(signal):
    """Computes mean value of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which mean is computed.
    Returns
    -------
    float
        Mean result
    """
    return np.mean(signal)



def cal_median(signal):
    """Computes median of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which median is computed
    Returns
    -------
    float
        Median result
    """
    return np.median(signal)


def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed
    Returns
    -------
    float
        Mean absolute deviation result
    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)



def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed
    Returns
    -------
    float
        Mean absolute deviation result
    """
    return scipy.stats.median_absolute_deviation(signal, scale=1)


def rms(signal):
    """Computes root mean square of the signal.
    Square root of the arithmetic mean (average) of the squares of the original values.
    Parameters
    ----------
    signal : nd-array
        Input from which root mean square is computed
    Returns
    -------
    float
        Root mean square
    """
    return np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


def cal_std(signal):
    """Computes standard deviation (std) of the signal.
    Parameters
    ----------
    signal : nd-array
        Input from which std is computed
    Returns
    -------
    float
        Standard deviation result
    """
    return np.std(signal)

def cal_var(signal):
    """Computes variance of the signal.
    
    Parameters
    ----------
    signal : nd-array
       Input from which var is computed
    Returns
    -------
    float
        Variance result
    """
    return np.var(signal)



def cal_fft(signal,fs):
   
    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()

def max_frequency(f,cum_fmag):
    """Computes maximum frequency of the signal.

    Parameters
    ----------
    cum_fmag: cumulative sum of amplitude of the frequency
    
    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
 

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]


def median_frequency(f,cum_fmag):
    """Computes median frequency of the signal.
    Parameters
    ----------
    cum_fmag: cumulative sum of amplitude of the frequency
    
    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]

    return f_median

def fundamental_frequency(signal, fs):
    """Computes fundamental frequency of the signal.
    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.
    
    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : int
        Sampling frequency
    Returns
    -------
    f0: float
       Predominant frequency of the signal
    """
    signal = signal - np.mean(signal)
    f, fmag = cal_fft(signal, fs)

    # Finding big peaks, not considering noise peaks with low amplitude

    bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

    # # Condition for offset removal, since the offset generates a peak at frequency zero
    bp = bp[bp != 0]
    if not list(bp):
        f0 = 0
    else:
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0   

def fea_fft(signal,fs):
    
    f,fmag = cal_fft(signal,fs)
    cum_fmag = np.cumsum(fmag)
    f_max = max_frequency(f,cum_fmag)
    f_median = median_frequency(f,cum_fmag)
    f_fundamental = fundamental_frequency(signal, 1/0.35)
    
    return f_max,f_median,f_fundamental
    
    
    
    
def feature_add(signal):
    T =0.35
    fs =1/T
    entropy_val = entropy(signal, prob='standard')
    var = cal_var(signal)
    std= cal_std(signal)
    root_mean_sqrt = rms(signal)
    max_data=cal_max(signal)
    min_data=cal_min(signal)
    mean_data = cal_mean(signal)
    median_data = cal_median(signal)
    d_mean= mean_abs_deviation(signal)
    d_median = median_abs_deviation(signal)
    f_max,f_median,f_fundamental = fea_fft(signal,fs)
    kurtosis = kurtosis(signal)
    skewness = skewness(signal)
    return [entropy_val,max_data,min_data,mean_data,median_data,d_mean,d_median,var,std,root_mean_sqrt,f_max,f_median,f_fundamental,kurtosis,skewness] 
   
data_files_all = pd.read_pickle("./DataSignal.pkl")
data_features = data_files_all[['Model','Status','Closing speed','Penetration']]

#Adding features for acc signal
A = []
for a in data_files_all['Acc']:
    l=[]
    l = feature_add(a)
    A.append(l)
    #d_acc=pd.DataFrame(l, columns=['Entropy','SpectralEntropy_A','max_A','min','mean','median','DevationMean','DevationMedian','var','std','rms'])

df_t= pd.DataFrame(A,columns=['Entropy_A','max_A','min_A','mean_A','median_A','DevationMean_A','DevationMedian_A','var_A','std_A','rms_A','f_max_A','f_median_A','f_fundamental_A','kurtosis_A','skewness_A'])
data_features_acc = pd.concat([data_features, df_t.reindex(data_features.index)], axis=1)    

#Adding features for vel signal

V = []
for a in data_files_all['Vel']:
    l=[]
    l = feature_add(a)
    V.append(l)
    #d_acc=pd.DataFrame(l, columns=['Entropy','SpectralEntropy_A','max_A','min','mean','median','DevationMean','DevationMedian','var','std','rms'])

data_features_vel= pd.DataFrame(V,columns=['Entropy_V','max_V','min_V','mean_V','median_V','DevationMean_V','DevationMedian_V','var_V','std_V','rms_V','f_max_V','f_median_V','f_fundamental_V','kurtosis_V','skewness_V'])

d_a_v = pd.concat([data_features_acc, data_features_vel.reindex(data_features_acc.index)], axis=1)

#Adding features for pos signal

P= []
for a in data_files_all['Pos']:
    l=[]
    l = feature_add(a)
    P.append(l)
    #d_acc=pd.DataFrame(l, columns=['Entropy','SpectralEntropy_A','max_A','min','mean','median','DevationMean','DevationMedian','var','std','rms'])

data_features_pos= pd.DataFrame(V,columns=['Entropy_P','max_P','min_P','mean_P','median_P','DevationMean_P','DevationMedian_P','var_P','std_P','rms_P','f_max_P','f_median_P','f_fundamental_P','kurtosis_P','skewness_P'])

data_features_all =pd.concat([d_a_v, data_features_pos.reindex(d_a_v.index)], axis=1)


file_name = 'feature.csv'
data_features_all.to_csv(file_name, sep=',')