import nptdms 
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
import torch
import copy
import scipy.signal


#Function to read TDMS files
def read_data_s(fname):
    """Reads acc, vel and pos of the signal from TDMS.
    Feature computational cost: 1
    Parameters
    ----------
    signal : .tdms file
        Input from which  acc, vel and pos is computed
    Returns
    -------
    arrays
      acc,vel,pos
    """
    calculated_data = pd.DataFrame()
    tdms_file = nptdms.TdmsFile(fname)
    calculated_data = tdms_file["calculated"].as_dataframe() 
    acc_reading = calculated_data["Acc.Z"].to_numpy()
    vel_reading = calculated_data["Velocity"].to_numpy()
    pos_reading = calculated_data["Position"].to_numpy()
   
    return np.asarray(acc_reading),np.asarray(vel_reading),np.asarray(pos_reading)

#Data loading
df_10_20 = pd.read_csv("/home/admin/Documents/master_thesis/usb/Messungen_G10_2020/DoCC_Messungen_Bewertungfiltered.csv",delimiter=',')
df_09_20 = pd.read_csv("/home/admin/Documents/master_thesis/usb/valid/DoCCMessung_G8_KW39/DoCC_Messungen_09_2020.csv",delimiter=',')
#df_info = df_row_merged = pd.concat([df_10_20, df_09_20], ignore_index=False)
add_1 = '/home/admin/Documents/master_thesis/usb/Messungen_G10_2020/'
add_2 = '/home/admin/Documents/master_thesis/usb/valid/DoCCMessung_G8_KW39/'

list_acc=[]
list_vel=[]
list_pos=[]
list_acc_10=[]
list_vel_10=[]
list_pos_10=[]

#extracting values from files
for name in df_09_20['File']:
    fname = add_2+name
    acc=[]
    vel=[]
    pos=[]

    acc,vel,pos = read_data_s(fname)
    
    list_acc.append(acc)
    list_vel.append(vel)
    list_pos.append(pos)

for name in df_10_20['File']:
    fname = add_1+name
    acc=[]
    vel=[]
    pos=[]

    acc,vel,pos = read_data_s(fname)

    list_acc_10.append(acc)
    list_vel_10.append(vel)
    list_pos_10.append(pos)
model = []
door =[]
for name in df_10_20['File']:
    model.append(name[25:28])
        
df_10_20['Model'] = model
df_10_20.drop(df_10_20[df_10_20.Model== 'F45'].index, inplace=True)
df_10_20=df_10_20.drop('Model',1)

df_09_20['Acc']= list_acc
df_09_20['Vel']= list_vel
df_09_20['Pos']= list_pos
df_10_20['Acc']= list_acc_10
df_10_20['Vel']= list_vel_10
df_10_20['Pos']= list_pos_10

#merging both datasets
df_info = df_row_merged = pd.concat([df_10_20, df_09_20], ignore_index=False)
model = []
door =[]
for name in df_info['File']:
    model.append(name[25:28])
    door.append(name[29:31])
        
df_info['Model'] = model 
df_info['Door'] = door  
data_files_all = df_info[['Messungid','Model','Door','Status','Closing speed','Penetration','Acc','Vel','Pos']]

#saving to pickle format
data_files_all.to_pickle("./DataSignal.pkl")