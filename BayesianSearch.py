# necessary imports
import os
import nibabel as nib
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from skopt import BayesSearchCV

# define a function to generate lists of paths for each scan
def getAllFilesinDir(path,resultList):
    filesList=os.listdir(path)
    for fileName in filesList:
        fileAbpath=os.path.join(path,fileName)
        if os.path.isdir(fileAbpath):
            getAllFilesinDir(fileAbpath,resultList)
        else:
            if fileName=='normInt_Vol_wmean.nii':
                resultList.append(fileAbpath)

# list of FTD scans
FTD_scan_paths  = []
dir_FTD = "data_CNN/FTD"
getAllFilesinDir(dir_FTD, FTD_scan_paths)

# list of AD scans
AD_scan_paths  = []
dir_AD = "data_CNN/AD"
getAllFilesinDir(dir_AD, AD_scan_paths)

# list of CN scans
CN_scan_paths  = []
dir_CN = "data_CNN/CN"
getAllFilesinDir(dir_CN, CN_scan_paths)

random.seed(5)
random.shuffle(FTD_scan_paths)
random.shuffle(AD_scan_paths)
random.shuffle(CN_scan_paths)

FTD_train_scans = FTD_scan_paths[:173]
AD_train_scans = AD_scan_paths[:179]
CN_train_scans = CN_scan_paths[:180]

FTD_test_scans = FTD_scan_paths[173:]
AD_test_scans = AD_scan_paths[179:]
CN_test_scans = CN_scan_paths[180:]

# define functions to read nifti files and feature-wise normalize nifti
def read_nifti(path):
    scan = nib.load(path)
    scan = scan.get_fdata()
    return scan

def type_float32(volume):
    volume = volume.astype("float32")
    return volume

def preprocess_scan(path):
    volume = read_nifti(path)
    volume = type_float32(volume)
    return volume
    
    
# transform each scan into an np array and remove the top 10 layers and the bottom 9 layers
train_FTD_scans = (np.array([preprocess_scan(path) for path in FTD_train_scans]))[:,:,:,10:70]
train_AD_scans = (np.array([preprocess_scan(path) for path in AD_train_scans]))[:,:,:,10:70]
train_CN_scans = (np.array([preprocess_scan(path) for path in CN_train_scans]))[:,:,:,10:70]

# assign labels to each category
train_FTD_labels = np.array([2 for _ in range(len(train_FTD_scans))])
train_AD_labels = np.array([1 for _ in range(len(train_AD_scans))])
train_CN_labels = np.array([0 for _ in range(len(train_CN_scans))])

X_train = np.concatenate((train_FTD_scans, train_AD_scans, train_CN_scans), axis=0)
y_train = np.concatenate((train_FTD_labels, train_AD_labels, train_CN_labels), axis=0)

# same for testing scans
test_FTD_scans = (np.array([preprocess_scan(path) for path in FTD_test_scans]))[:,:,:,10:70]
test_AD_scans = (np.array([preprocess_scan(path) for path in AD_test_scans]))[:,:,:,10:70]
test_CN_scans = (np.array([preprocess_scan(path) for path in CN_test_scans]))[:,:,:,10:70]

test_FTD_labels = np.array([2 for _ in range(len(test_FTD_scans))])
test_AD_labels = np.array([1 for _ in range(len(test_AD_scans))])
test_CN_labels = np.array([0 for _ in range(len(test_CN_scans))])

X_test = np.concatenate((test_FTD_scans, test_AD_scans, test_CN_scans), axis=0)
y_test = np.concatenate((test_FTD_labels, test_AD_labels, test_CN_labels), axis=0)

# one-hot encoding
#y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# use the bayessearchcv function from the skopt library to determine efficient sets of hyperparameters
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

X_train = X_train.reshape(-1, 79, 95, 60, 1)

# create a model based on the VGG16 architecture
def create_model(learning_rate=0.0003, optimizer ='adagrad', dropout=0.4, initializer = 'glorot_uniform'):
  model = Sequential()
  model.add(Conv3D(filters = 8, kernel_size = 3, input_shape=(79, 95, 60, 1), activation = 'relu', kernel_initializer=initializer))
  model.add(Conv3D(filters = 16, kernel_size = 3, activation = 'relu', kernel_initializer=initializer))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Conv3D(filters = 32, kernel_size = 3, activation = 'relu', kernel_initializer=initializer))
  model.add(Conv3D(filters = 64, kernel_size = 3, activation = 'relu', kernel_initializer=initializer))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(units=4096, activation='relu', kernel_initializer=initializer))
  model.add(Dropout(dropout))

  model.add(Dense(units=1024, activation='relu', kernel_initializer=initializer))
  model.add(Dropout(dropout))

  model.add(Dense(units=3, activation='softmax'))
  model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

# wrap the model into a keras classifier to use bayessearchcv
keras_class = KerasClassifier(build_fn=create_model, epochs=150, batch_size=4, verbose=1)

# large ranges of hyperparameters were explored
parameters = {
    "learning_rate": [0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003, 0.0001],
    "optimizer": ['adam', 'adadelta', 'adagrad'],
    "batch_size": [4, 6, 8, 10, 12],
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "initializer": ['glorot_uniform', 'he_uniform', 'glorot_normal', 'he_normal']
}

# define an early stopping callback
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)

start_time = time.time()
cv = KFold(n_splits=2, shuffle=True)
bayes_search = BayesSearchCV(keras_class, parameters, n_iter = 20, cv=cv)
bayes_search.fit(X_train, y_train, validation_split = 0.2, callbacks = [early_stopping])
with open('time.txt', 'w') as f:
    f.write("--- %s seconds ---" % (time.time() - start_time))

# save results to a dataframe
df = pd.DataFrame(bayes_search.cv_results_)
df.to_csv('cv_results.csv', index = False)
