# necessary imports
import os, random, copy, math
import nibabel as nib
import numpy as np
import tensorflow as tf
import skimage.transform
from datetime import datetime
from tensorflow.keras.utils import to_categorical
import shutil
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# concatenante train and test
scan_paths = FTD_scan_paths + AD_scan_paths + CN_scan_paths
X = (np.array([preprocess_scan(path) for path in scan_paths]))[:,:,:,10:70]

FTD_labels = np.array([2 for _ in range(len(FTD_scan_paths))])
AD_labels = np.array([1 for _ in range(len(AD_scan_paths))])
CN_labels = np.array([0 for _ in range(len(CN_scan_paths))])

y = np.concatenate((FTD_labels, AD_labels, CN_labels), axis=0)
y = to_categorical(y)

model = tf.keras.models.load_model('my_model.h5')

import umap
from tensorflow.keras.models import Model # Import the Model class
from tensorflow.keras.layers import Flatten

# Extract the outputs of the last layer for a given dataset
last_layer_output = model.layers[-1].output
last_layer_model = Model(inputs=model.input, outputs=last_layer_output)
last_layer_output = last_layer_model.predict(X)

# Use UMAP to reduce the dimensionality of the flattened output of the last layer
umap_model = umap.UMAP(n_components=2)
umap_output = umap_model.fit_transform(last_layer_output)

y_int = np.argmax(y, axis=1)

labels = ["CN", "AD", "FTD"]

# Create a colormap for the labels
colors = ['red', 'green', 'blue']
cmap = matplotlib.colors.ListedColormap(colors)

fig, ax = plt.subplots(dpi=2000)

# Plot each class separately with a different color

for i, label in enumerate(np.unique(y_int)):
    mask = (y_int == label)
    ax = plt.subplot(111)
    ax.scatter(umap_output[mask, 0], umap_output[mask, 1], c=cmap(i), label=labels[i], s=15, alpha = 0.4)
    ax.set_title("(A) UMAP visualization by class", y=-0.07, pad=-25, verticalalignment="top", fontsize = 18)
    
# Add a legend with the class labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [1,2,0]

#add legend to plot
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right') 
plt.xlim([-12, 22])
plt.xlabel("First dimension", fontsize = 12)
plt.ylabel("Second dimension", fontsize = 12)
ax.spines[['right', 'top']].set_visible(False)
plt.savefig("UMAPDiseases_All.png", bbox_inches="tight", dpi = 2000)
plt.show()

# create list of center for each file

# FTD

FTD_scans = FTD_train_scans + FTD_test_scans
#FTD_scans = FTD_test_scans
center_FTD = []
for file in FTD_scans:
    if "NIFD" in file:
        center_FTD.append("FTLDNI")
    else:
        center_FTD.append("ULille")

# AD

AD_scans = AD_train_scans + AD_test_scans
#AD_scans = AD_test_scans
center_AD = []
for file in AD_scans:
    if "ADUHD" in file:
        center_AD.append("ULille")
    else:
        center_AD.append("ADNI")
    
# CN
import pandas as pd
data = pd.read_excel('/Users/antoinerogeau/Desktop/MD thesis/media/demographics.xlsx', sheet_name='CN')

CN_scans = CN_train_scans + CN_test_scans
#CN_scans = CN_test_scans
center_CN = []
for file in CN_scans:
    row = data.loc[data['FileName'] == file] # get row that matches the filename
    center = row['center'].values[0] # get the value of center from the row
    center_CN.append(center)
    center_CN = [x.replace('FTLDNI_MAYO', 'FTLDNI') for x in center_CN]
    
centers = center_FTD + center_AD + center_CN

# Create a colormap for the labels
colors = ['darkblue', 'crimson', 'lightgreen']
cmap = matplotlib.colors.ListedColormap(colors)

# Convert labels to numeric values
le = LabelEncoder()
labels = le.fit_transform(centers)

fig, ax = plt.subplots(dpi=2000)

# Plot each class separately with a different color
for i, label in enumerate(np.unique(labels)):
    mask = (labels == label)
    ax = plt.subplot(111)
    ax.scatter(umap_output[mask, 0], umap_output[mask, 1], c=cmap(i), label=le.inverse_transform([label])[0], s=15, alpha = 0.4)
    ax.set_title("(B) UMAP visualization by center", y=-0.07, pad=-25, verticalalignment="top", fontsize = 18)
    
# Add a legend with the class labels
plt.legend(loc='upper right')
plt.xlim([-12, 22])
plt.xlabel("First dimension", fontsize = 12)
plt.ylabel("Second dimension", fontsize = 12)
ax.spines[['right', 'top']].set_visible(False)
plt.savefig("UMAPCenters_All.png", bbox_inches="tight", dpi = 2000)
plt.show()
