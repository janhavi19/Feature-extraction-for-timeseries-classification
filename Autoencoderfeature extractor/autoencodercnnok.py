import pandas as pd
import numpy as np
import glob
from tensorflow.keras import layers, models, Model, optimizers
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.pyplot as plt
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import keras

from keras.models import Sequential, Model

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization,Dense
from keras.layers import Conv2D, UpSampling2D, MaxPool2D, Flatten, BatchNormalization

from keras.optimizers import SGD


from sklearn.utils import shuffle
from tensorflow.python.keras import layers, models, Model, optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models, Model, optimizers
#from keras.utils import np_utils, to_categorical
from random import randint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix 
import seaborn as sns
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.utils import class_weight

def load_data(cmodel,signal_name):
    path_OK = './image_sp/'+cmodel+'/'+signal_name+'/OK/*.*'
    path_NOK = './image_sp/'+cmodel+'/'+signal_name+'/NOK/*.*'
    OK = glob.glob(path_OK)
    NOK = glob.glob(path_NOK)
    data = []
    labels = []
    data_ok =[]
    lable_ok=[]

    for i in OK:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
        target_size= (32,32))
        image=np.array(image)
        data.append(image)
        data_ok.append(image)
        lable_ok.append('OK')
        labels.append('OK')
    for i in OK:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
        target_size= (32,32))
        image=np.array(image)
        data_ok.append(image)
        lable_ok.append('OK')
       
    for i in NOK:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
        target_size= (32,32))
        image=np.array(image)
        data.append(image)
        labels.append('NOK')

    data = np.array(data)
    labels = np.array(labels)
    data_ok = np.array(data_ok)
    lable_ok = np.array(lable_ok)

    X = data
    y = labels
    X_ok = data_ok
    y_ok = lable_ok

    X  = X .astype('float32')
    X /= 255
    X_ok  = X_ok.astype('float32')
    X_ok /= 255


    lb = LabelEncoder()
    y = lb.fit_transform(y)
    y_ok = lb.fit_transform(y_ok)

    return X,y,X_ok,y_ok

def create_block(input, chs): ## Convolution block of 2 layers
    x = input
    for i in range(2):
        x = Conv2D(chs, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x

def classifier_conv(inp):
    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))
    x = Conv2D(1024, 3, padding="same")(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, 3, padding="same")(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.69)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(input, output)

def autoencoder_block():
    input = Input((32,32,3))
    
    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    
    #Middle
    x = MaxPool2D(2)(block2)
    middle = create_block(x, 128)
    
    # Decoder
    block3 = create_block(middle, 64)
    up1 = UpSampling2D((2,2))(block3)
    block4 = create_block(up1, 32)
    up2 = UpSampling2D((2,2))(block4)
    
    # output
    x = Conv2D(3, 1)(up2)
    output = Activation("sigmoid")(x)
    return Model(input, middle), Model(input, output)

def feature_extractor(X,y,X_test,y_test,cmodel,signal_type):
    

    encoder, model = autoencoder_block()
    model.compile(SGD(1e-3, 0.9), loss='mse')
    history = model.fit(X, X, 
                       batch_size=512,
                       epochs=100,
                       verbose=1,
                       shuffle=True)

    model.save('./result/'+cmodel+'/'+signal_type+'_autoencoder.h5') 
    features_train = encoder.predict(X)
    features_test = encoder.predict(X_test)
    
    return (features_train,y, features_test,y_test)

def conf_plot(error_df,model,signal_type):

    LABELS = ["OK","NOK"]

    conf_matrix = confusion_matrix(error_df.True_class, error_df.Predictions)
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('./result/'+model+'/'+signal_type+'_automodel.png')


def classification( X_f_train,X_f_test,y_f_train,y_f_test,cmodel,signal_type):
    print(X_f_train.shape)
    print(X_f_test.shape) 
   

    modelcnn = classifier_conv(X_f_train)
    modelcnn.compile(loss="binary_crossentropy", optimizer='adam',metrics=['accuracy'])

    hist1 = modelcnn.fit(X_f_train, y_f_train, batch_size=512, epochs=50, 
                            validation_data = (X_f_test, y_f_test),
                            shuffle=True)

    
    modelcnn.save('./result/'+cmodel+'/'+signal_type+'_resultautoclass.h5') 
    y_pred =  modelcnn.predict(X_f_test)
    
    threshold_fixed = 2.9843744e-25
    y_pred = y_pred.flatten()
    y_pred = y_pred.tolist()

    pred_y = [1 if e > threshold_fixed else 0 for e in y_pred]


    loss, accuracy=  modelcnn.evaluate(X_f_test,y_f_test)
    
    error_df = pd.DataFrame({'Predictions':pred_y,
                         'True_class': y_f_test})
    print(error_df)
    conf_plot(error_df,cmodel,signal_type)

    print('\nLoss: %.2f, Accuracy: %.2f%%' %(loss,accuracy*100) )
    

if __name__ == "__main__":
    cmodel,signal_type = str(sys.argv[1]), str(sys.argv[2])
    X,y,X_ok,y_ok = load_data(cmodel,signal_type)
    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle= True)
    X_features_train,y_features_train,X_f_test,y_f_test = feature_extractor(X_ok,y_ok,X_test,y_test,cmodel,signal_type)
    
    classification( X_features_train,X_f_test,y_features_train,y_f_test,cmodel,signal_type)
    

