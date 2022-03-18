import numpy as np
import pandas as pd
import pylab
import sys
import matplotlib as plt

plt.rcParams.update({'figure.max_open_warning': 0})

def get_model_data(data,name):
    df_model =data.loc[((data['Status'] == "OK")|(data['Status'] == "NOK"))&(data['Model'] == name)]
    return df_model

def graph_spectrogram(data_signal,index,model,signal_type,T,N):
    
    fs = N/T
    
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    
    pylab.specgram(data_signal, Fs=fs)
    pylab.savefig('./image_sp/'+model+'/'+signal_type+'/'+'spectrogram'+model+'_'+str(index)+'.png')

def conv_array(signal):
    result=[]
    for r in signal:
        result.append(r)
    return np.asarray(result)

def spectogram_generator(data,model,signal_type):
    df = get_model_data(data,model)
    signal_array = conv_array(df[signal_type])
    for idx, signal in enumerate(signal_array):
        N = len(signal)
        T = N*0.001
        graph_spectrogram(signal,idx,model,signal_type,T,N)




    
if __name__ == "__main__":
    data = pd.read_pickle("./DataSignal.pkl")
     
    spectogram_generator(data,str(sys.argv[1]), str(sys.argv[2]))


    


