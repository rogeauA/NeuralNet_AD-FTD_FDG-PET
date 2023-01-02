# necessary imports
import os, random, copy, math
import nibabel as nib
import numpy as np
import tensorflow as tf
import skimage.transform
from datetime import datetime
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

model = tf.keras.models.load_model('final_model.h5')

# test with list of predictions
# in order to shorten computing time, we used tensorflow ability to predict on multiple volumes at once
# to do this, arrays are concatenated with the moving occluding cube, until reaching a "size_array_optimal"
# we then predicted and predictions are added to the heatmap for each class
# an optimal array size of 18 was found best efficient for our computer setup 
def OcclusionFTD(list_of_data, occluding_stride, size_array_optimal = 18):
    global dic
    dic = {}
    global native_dic
    native_dic = {}
    for volume in list_of_data:
        print('scanning position (%s)'%(volume))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        volume = preprocess_scan(volume)
        volume = np.expand_dims(volume[:,:,10:70], axis = 0)
        volume = np.expand_dims(volume, axis = 4)
        # Find the index of each class
        index_object = 2
        _, height, width, depth, _ = volume.shape
        output_height = int(math.ceil((height-5) / occluding_stride + 1))
        output_width = int(math.ceil((width-5) / occluding_stride + 1))
        output_depth = int(math.ceil((depth-5) / occluding_stride + 1))
        heatmap = np.empty((output_height, output_width, output_depth))
        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        number = -1
        for h in range(output_height):
            for w in range(output_width):
                for d in range(output_depth):
                    # Occluder region:
                    h_start = h * occluding_stride
                    w_start = w * occluding_stride
                    d_start = d * occluding_stride
                    h_end = min(height, h_start + 5)
                    w_end = min(width, w_start + 5)
                    d_end = min(depth, d_start + 5)
                    # Getting the image copy, applying the occluding window and classifying it again:
                    input_volume = copy.copy(volume)
                    input_volume[:, h_start:h_end, w_start:w_end, d_start:d_end, :] =  0
                    #input_volume = np.expand_dims(input_volume, axis = 0)
                    arr = np.concatenate((arr, input_volume), axis = 0)
                    if arr.shape[0] == size_array_optimal:
                        number = number + 1
                        out = model.predict(arr, verbose = 0)
                        out = out[:,2]
                        for i, value in enumerate(out):
                            overall_i = number * size_array_optimal + i
                            position_h = overall_i // (output_width * output_depth)
                            rest = overall_i % (output_width * output_depth)
                            position_w = rest // output_depth
                            position_d = rest % (output_depth)
                            heatmap[position_h, position_w, position_d] = value
                        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        if arr.shape[0] > 0:
            out = model.predict(arr, verbose = 0)
            out = out[:,2]
            number = number + 1
            for i, value in enumerate(out):
                overall_i = number * size_array_optimal + i
                position_h = overall_i // (output_width * output_depth)
                rest = overall_i % (output_width * output_depth)
                position_w = rest // output_depth
                position_d = rest % output_depth
                heatmap[position_h, position_w, position_d] = value
        heatmap = heatmap / (np.amax(heatmap))
        native_dic["string_{}".format(str(volume))] = heatmap
        heatmap2 = skimage.transform.resize(heatmap, (75, 91, 56), order = 3)
        dic["string_{}".format(str(volume))] = heatmap2
    list_of_NativeHeatmaps = list(native_dic.values())
    global NativeHeatmap_FTD
    NativeHeatmap_FTD = np.mean((list_of_NativeHeatmaps), axis=0)
    np.save('NativeHeatmap_FTD{}.npy'.format(occluding_stride), NativeHeatmap_FTD)
    list_of_heatmaps = list(dic.values())
    global heatmap_FTD
    heatmap_FTD = np.mean((list_of_heatmaps), axis=0)
    np.save('heatmap_FTD{}.npy'.format(occluding_stride), heatmap_FTD)
    
def OcclusionAD(list_of_data, occluding_stride, size_array_optimal = 18):
    global dic
    dic = {}
    global native_dic
    native_dic = {}
    for volume in list_of_data:
        print('scanning position (%s)'%(volume))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        volume = preprocess_scan(volume)
        volume = np.expand_dims(volume[:,:,10:70], axis = 0)
        volume = np.expand_dims(volume, axis = 4)
        # Find the index of each class
        index_object = 1
        _, height, width, depth, _ = volume.shape
        output_height = int(math.ceil((height-5) / occluding_stride + 1))
        output_width = int(math.ceil((width-5) / occluding_stride + 1))
        output_depth = int(math.ceil((depth-5) / occluding_stride + 1))
        heatmap = np.empty((output_height, output_width, output_depth))
        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        number = -1
        for h in range(output_height):
            for w in range(output_width):
                for d in range(output_depth):
                    # Occluder region:
                    h_start = h * occluding_stride
                    w_start = w * occluding_stride
                    d_start = d * occluding_stride
                    h_end = min(height, h_start + 5)
                    w_end = min(width, w_start + 5)
                    d_end = min(depth, d_start + 5)
                    # Getting the image copy, applying the occluding window and classifying it again:
                    input_volume = copy.copy(volume)
                    input_volume[:, h_start:h_end, w_start:w_end, d_start:d_end, :] =  0
                    #input_volume = np.expand_dims(input_volume, axis = 0)
                    arr = np.concatenate((arr, input_volume), axis = 0)
                    if arr.shape[0] == size_array_optimal:
                        number = number + 1
                        out = model.predict(arr, verbose = 0)
                        out = out[:,1]
                        for i, value in enumerate(out):
                            overall_i = number * size_array_optimal + i
                            #print(overall_i)
                            #print(value)
                            position_h = overall_i // (output_width * output_depth)
                            rest = overall_i % (output_width * output_depth)
                            position_w = rest // output_depth
                            position_d = rest % (output_depth)
                            heatmap[position_h, position_w, position_d] = value
                        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        if arr.shape[0] > 0:
            out = model.predict(arr, verbose = 0)
            out = out[:,1]
            number = number + 1
            for i, value in enumerate(out):
                overall_i = number * size_array_optimal + i
                position_h = overall_i // (output_width * output_depth)
                rest = overall_i % (output_width * output_depth)
                position_w = rest // output_depth
                position_d = rest % output_depth
                heatmap[position_h, position_w, position_d] = value
        heatmap = heatmap / (np.amax(heatmap))
        native_dic["string_{}".format(str(volume))] = heatmap
        heatmap2 = skimage.transform.resize(heatmap, (75, 91, 56), order = 3)
        dic["string_{}".format(str(volume))] = heatmap2
    list_of_NativeHeatmaps = list(native_dic.values())
    global NativeHeatmap_FTD
    NativeHeatmap_FTD = np.mean((list_of_NativeHeatmaps), axis=0)
    np.save('NativeHeatmap_AD{}.npy'.format(occluding_stride), NativeHeatmap_FTD)
    list_of_heatmaps = list(dic.values())
    global heatmap_AD
    heatmap_AD = np.mean((list_of_heatmaps), axis=0)
    np.save('heatmap_AD{}.npy'.format(occluding_stride), heatmap_AD)
    
def OcclusionCN(list_of_data, occluding_stride, size_array_optimal = 18):
    global dic
    dic = {}
    global native_dic
    native_dic = {}
    for volume in list_of_data:
        print('scanning position (%s)'%(volume))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        volume = preprocess_scan(volume)
        volume = np.expand_dims(volume[:,:,10:70], axis = 0)
        volume = np.expand_dims(volume, axis = 4)
        # Find the index of each class
        index_object = 0
        _, height, width, depth, _ = volume.shape
        output_height = int(math.ceil((height-5) / occluding_stride + 1))
        output_width = int(math.ceil((width-5) / occluding_stride + 1))
        output_depth = int(math.ceil((depth-5) / occluding_stride + 1))
        heatmap = np.empty((output_height, output_width, output_depth))
        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        number = -1
        for h in range(output_height):
            for w in range(output_width):
                for d in range(output_depth):
                    # Occluder region:
                    h_start = h * occluding_stride
                    w_start = w * occluding_stride
                    d_start = d * occluding_stride
                    h_end = min(height, h_start + 5)
                    w_end = min(width, w_start + 5)
                    d_end = min(depth, d_start + 5)
                    # Getting the image copy, applying the occluding window and classifying it again:
                    input_volume = copy.copy(volume)
                    input_volume[:, h_start:h_end, w_start:w_end, d_start:d_end, :] =  0
                    #input_volume = np.expand_dims(input_volume, axis = 0)
                    arr = np.concatenate((arr, input_volume), axis = 0)
                    if arr.shape[0] == size_array_optimal:
                        number = number + 1
                        out = model.predict(arr, verbose = 0)
                        out = out[:,0]
                        for i, value in enumerate(out):
                            overall_i = number * size_array_optimal + i
                            #print(overall_i)
                            #print(value)
                            position_h = overall_i // (output_width * output_depth)
                            rest = overall_i % (output_width * output_depth)
                            position_w = rest // output_depth
                            position_d = rest % (output_depth)
                            heatmap[position_h, position_w, position_d] = value
                        arr = np.empty((0, 79, 95, 60, 1), dtype=np.float32)
        if arr.shape[0] > 0:
            out = model.predict(arr, verbose = 0)
            out = out[:,0]
            number = number + 1
            for i, value in enumerate(out):
                overall_i = number * size_array_optimal + i
                position_h = overall_i // (output_width * output_depth)
                rest = overall_i % (output_width * output_depth)
                position_w = rest // output_depth
                position_d = rest % output_depth
                heatmap[position_h, position_w, position_d] = value
        heatmap = heatmap / (np.amax(heatmap))
        native_dic["string_{}".format(str(volume))] = heatmap
        heatmap2 = skimage.transform.resize(heatmap, (75, 91, 56), order = 3)
        dic["string_{}".format(str(volume))] = heatmap2
    list_of_NativeHeatmaps = list(native_dic.values())
    global NativeHeatmap_CN
    NativeHeatmap_CN = np.mean((list_of_NativeHeatmaps), axis=0)
    np.save('NativeHeatmap_CN{}.npy'.format(occluding_stride), NativeHeatmap_CN)
    list_of_heatmaps = list(dic.values())
    global heatmap_CN
    heatmap_CN = np.mean((list_of_heatmaps), axis=0)
    np.save('heatmap_CN{}.npy'.format(occluding_stride), heatmap_CN)
    
OcclusionFTD(FTD_train_scans, 2, 18)
OcclusionAD(AD_train_scans, 2, 18)
OcclusionCN(CN_train_scans, 2, 18)
