# necessary imports
import os, time, random
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage, interp
from itertools import cycle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

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

# define dataset augmentation functions
@tf.function

# fliping along the sagittal plan in 50% of cases
def flip(volume):
    chance = random.uniform(0,1)
    if chance > 0.5:
        volume = K.reverse(volume, axes=0)
    else:
        volume = volume
    return volume

# rotation between -10 and 10° in each plan in 50% of cases to minimize computing time
def rotate(volume):
    def scipy_rotate(volume):
        # pick angles at random
        angle0 = random.randint(-10, 10)
        angle1 = random.randint(-10, 10)
        angle2 = random.randint(-10, 10)
        # rotate volume
        if random.uniform(0,1) > 0.5: 
            volume = ndimage.rotate(volume, angle0, axes=(1, 0), reshape=False)
        if random.uniform(0,1) > 0.5: 
            volume = ndimage.rotate(volume, angle1, axes=(2, 0), reshape=False)
        if random.uniform(0,1) > 0.5: 
            volume = ndimage.rotate(volume, angle2, axes=(1, 2), reshape=False)
        else:
            volume = volume
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return np.ndarray.astype(volume, np.float32)
    
    volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return volume

# translation along each axis in 100% of cases
def shift(volume):
    def scipy_shift(volume):
        # define random shifts with a maximum = 10% in each axis
        sequence = (random.randint(-8, 8), random.randint(-10, 10), random.randint(-6, 6))
        # shift volume
        volume = ndimage.shift(volume, sequence)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return np.ndarray.astype(volume, np.float32)
    
    volume = tf.numpy_function(scipy_shift, [volume], tf.float32)
    return volume

# augmentation function for train dataset loader
def augment(volume, label):
    volume = flip(volume)
    volume = shift(volume)
    volume = rotate(volume)
    tf.keras.backend.set_floatx('float32')
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# validation function for train dataset loader
def val_preprocess(volume, label):
    tf.keras.backend.set_floatx('float32')
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
  
# based on the results of the bayesian optimized search we used the following hyperparameters
def create_model(learning_rate=0.0005, optimizer ='adagrad', dropout = 0.5):
  model = Sequential()
  model.add(Conv3D(filters = 8, kernel_size = 3, input_shape=(79, 95, 60, 1), activation = 'relu'))
  model.add(Conv3D(filters = 16, kernel_size = 3, activation = 'relu'))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Conv3D(filters = 32, kernel_size = 3, activation = 'relu'))
  model.add(Conv3D(filters = 64, kernel_size = 3, activation = 'relu'))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(units=4096, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(units=1024, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(units=3, activation='softmax'))
  model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

BATCH_SIZE = 6

# Set random seeds for repeatable results
np.random.seed(5)
tf.random.set_seed(5)

# create 5 folds in the dataset
kf = StratifiedKFold(n_splits = 5, shuffle = True)

AUTOTUNE = tf.data.AUTOTUNE

validation_accuracy = []
validation_loss = []

# set 1st fold and create directory to save models
%matplotlib inline
fold = 1
directory = 'saved_models/'

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

# set timer
start_time = time.time()

# over the 5 folds, augment training data and test on native validation data
for train_index, test_index in kf.split(X_train, y_train):
    X_train_split, X_val = X_train[train_index], X_train[test_index]
    y_train_split, y_val = y_train[train_index], y_train[test_index]
    
    #one hot encoding
    y_train_split = to_categorical(y_train_split)
    y_val = to_categorical(y_val)
    
    train_loader = tf.data.Dataset.from_tensor_slices((X_train_split, y_train_split))
    train_dataset = (
    train_loader.cache()
    .shuffle(len(X_train_split))
    .map(augment, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))
    
    val_loader = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = (
    val_loader.cache()
    .shuffle(len(X_val))
    .map(val_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))
    
    # Create a new model instance 
    model = create_model()
    
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)
    save_best = ModelCheckpoint(directory + get_model_name(fold), monitor = 'val_accuracy', verbose = 1, save_best_only = True)
    
    # train the model on the current fold
    history=model.fit(train_dataset, epochs=150, validation_data=val_dataset, verbose=2, callbacks=[early_stopping, save_best])
    model.load_weights('saved_models/model_'+str(fold)+'.h5')
    
    results = model.evaluate(val_dataset)
    results = dict(zip(model.metrics_names,results))

    validation_accuracy.append(results['accuracy'])
    validation_loss.append(results['loss'])

    # show accuracy plots
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_{}.png'.format(fold))
    
    # show loss plots
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_{}.png'.format(fold))
    
    # clear and jump to the next fold
    tf.keras.backend.clear_session()

    fold += 1

# return time to train the 5 models
with open('time.txt', 'w') as f:
    f.write("--- %s seconds ---" % (time.time() - start_time))
    
# select the best model on validation accuracy and retrain it on the whole non-augmented dataset
X_train = np.concatenate((train_FTD_scans, train_AD_scans, train_CN_scans), axis=0)
y_train = np.concatenate((train_FTD_labels, train_AD_labels, train_CN_labels), axis=0)
y_train = to_categorical(y_train)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 6

total_train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
total_train_dataset = (total_train_loader.cache()
    .shuffle(len(X_train))
    .map(val_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE))

sgd = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)

def create_model(dropout = 0.5):
  model = Sequential()
  model.add(Conv3D(filters = 8, kernel_size = 3, input_shape=(79, 95, 60, 1), activation = 'relu'))
  model.add(Conv3D(filters = 16, kernel_size = 3, activation = 'relu'))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Conv3D(filters = 32, kernel_size = 3, activation = 'relu'))
  model.add(Conv3D(filters = 64, kernel_size = 3, activation = 'relu'))
  model.add(MaxPool3D(pool_size=3))
  model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(units=4096, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(units=1024, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(units=3, activation='softmax'))

  model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return model

model = create_model()

# in our case the best model from KFold training was model 1
model.load_weights('saved_models/model_'+str(1)+'.h5')
model.fit(total_train_dataset, epochs=50)
model.save_weights("final_model.h5")
