# necessary imports
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from skimage.transform import resize
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

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

FTD_scans = (np.array([preprocess_scan(path) for path in FTD_scan_paths]))[:,:,:,10:70]
AD_scans = (np.array([preprocess_scan(path) for path in AD_scan_paths]))[:,:,:,10:70]
CN_scans = (np.array([preprocess_scan(path) for path in CN_scan_paths]))[:,:,:,10:70]

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

model = tf.keras.models.load_model('my_model.h5')
model.summary()

#model.summary()
###---lAYER-Name--to-visualize--###
LAYER_NAME='conv3d_31'
# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])
grad_model.summary()

dic_cam_maps = {}

for i in range(AD_scans.shape[0]):
    io_img=tf.expand_dims(AD_scans[i], axis=-1)
    io_img=tf.expand_dims(io_img, axis=0)
    print(i)
    ###----index of the class (class 1 = AD here)
    CLASS_INDEX=1

    ###--Compute GRADIENT
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(io_img)
        loss = predictions[:, CLASS_INDEX]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1,2))
    # Build a ponderated map of filters according to gradients importance
    cam = np.zeros(output.shape[0:3], dtype=np.float32)

    cam = np.array(cam)

    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    cam = np.array(cam)
    max_voxl = np.max(cam)
    Nz_cam = cam/max_voxl
    key_name = f'cam_{i}'
    dic_cam_maps[key_name] = Nz_cam

list_of_cam = list(dic_cam_maps.values())
cam_AD = np.mean((list_of_cam), axis=0)

# mask on the brain
mean = preprocess_scan('meanimage/AD/meanAll_normInt_Vol_wmean.nii')[:, :, 10:70]
mean_mask = np.where(mean != 0, 1, mean)

capi=resize(cam_AD, (79, 95, 60))
capi = capi*mean_mask

#print(capi.shape)
capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 18))
for i, ax in enumerate(axes.flat):
    #im = ax.imshow(np.rot90(mean[:, :, (i+1)*2+3]), cmap = 'Greys')
    #heatmap_im = ax.imshow(np.rot90(heatmap[:, :, (i+1)*2+5]), cmap = 'jet', alpha = 0.6)
    axial_ct_img=np.squeeze(np.rot90(mean[:, :,i*2]))
    axial_grad_cmap_img=np.squeeze(np.rot90(heatmap[:,:, i*2]))
    axial_overlay=cv2.addWeighted(axial_ct_img,0.3,axial_grad_cmap_img, 0.6, 0)
    img_plot = ax.imshow(axial_overlay,cmap='jet', vmin=0.05, vmax=0.55)
    
[axi.set_axis_off() for axi in axes.ravel()]
plt.subplots_adjust(wspace=-0.14, hspace=0, left=0, right=1, bottom=0, top=1)
fig.colorbar(img_plot, ax=axes.ravel().tolist(), fraction=0.019, pad=0.01, location = 'bottom', aspect=40)
#plt.savefig('CAM_AD.png', bbox_inches = "tight") 
plt.show()
