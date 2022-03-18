import pandas as pd
import shutil
import sys

def move_to(model,signal_name,status,df):
    df_model=df.loc[((df['Status'] == "OK")|(df['Status'] == "NOK"))&(df['Model'] == model)]

    name = []
    for index,a in enumerate(df_model['Status']):
        s= 'spectrogram'+model+'_'+str(index)+'.png'
        name.append(s)

    df_model['SpectogramPath']=name

    df_OK = df_model.loc[(df_model['Status']=="OK")]
    df_NOK = df_model.loc[(df_model['Status']=="NOK")]

    destination =  './image_sp/'+model+'/'+signal_name+'/'+status+'/'
    source = './image_sp/'+model+'/'+signal_name+'/'

    if  status == 'OK':

        for f in df_OK['SpectogramPath']:
            src = source+f
            dst = destination
            shutil.move(src,dst)
    
    else:
        for f in df_NOK['SpectogramPath']:
            src = source+f
            dst = destination
            shutil.move(src,dst)


    


if __name__ == "__main__":

    data = pd.read_pickle("./DataSignal.pkl")
     
    move_to(str(sys.argv[1]),str(sys.argv[2]),'OK',data)
    move_to(str(sys.argv[1]),str(sys.argv[2]),'NOK',data)

