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
import itertools

import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv1D, MaxPooling2D, AveragePooling1D
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Dense, Embedding, LSTM
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def load_data(cmodel,signal_name):
    path_OK = './image_sp/'+cmodel+'/'+signal_name+'/OK/*.*'
    path_NOK = './image_sp/'+cmodel+'/'+signal_name+'/NOK/*.*'
    OK = glob.glob(path_OK)
    NOK = glob.glob(path_NOK)
    data = []
    labels = []
    for i in OK:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
        target_size= (224,224))
        image=np.array(image)
        data.append(image)
        labels.append('OK')
    for i in NOK:
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
        target_size= (224,224))
        image=np.array(image)
        data.append(image)
        labels.append('NOK')

    data = np.array(data)
    labels = np.array(labels)

    X = data
    y = labels
    X = X.astype('float32')
    X /= 255

    lb = LabelEncoder()
    y = lb.fit_transform(y)
    
    return X,y

def conf_plot(error_df,model,signal_type):

    LABELS = ["NOK","OK"]

    conf_matrix = confusion_matrix(error_df.True_class, error_df.Predictions)
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('./results/'+model+'/'+signal_type+'_vggmodel.png')


def train(X,y,model,signal_type):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,shuffle=True, random_state=None)

    vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

    for layer in vgg_model.layers:
        layer.trainable = False

    x = vgg_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x) # Sigmoid
    transfer_model = Model(inputs=vgg_model.input, outputs=x)


    transfer_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    history = transfer_model.fit(X_train, y_train, batch_size = 1, epochs=50, validation_data=(X_test,y_test))
    transfer_model.save('./results/'+model+'/'+signal_type+'image_vgg.h5')

    y_pred = transfer_model.predict(X_test)
    y_pred= tf.argmax(y_pred,-1)



    loss, accuracy= transfer_model.evaluate(X_test,y_test)
    
    error_df = pd.DataFrame({'Predictions':y_pred,
                         'True_class': y_test})
    conf_plot(error_df,model,signal_type)

    print('\nLoss: %.2f, Accuracy: %.2f%%' %(loss,accuracy*100) )


if __name__ == "__main__":
    model,signal_type = str(sys.argv[1]), str(sys.argv[2])
    X,y = load_data(model,signal_type)
    train(X,y,model,signal_type)


