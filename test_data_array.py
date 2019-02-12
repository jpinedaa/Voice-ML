import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from LoadData import LoadData
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import argparse
import time


data = np.memmap('data.array', dtype= np.float64, mode= 'r+', shape= (250000,201,300,1))
labels = np.empty((250000,),dtype= np.unicode_)

print("shape: " + str(data.shape) )

print("data[0]:  ", data[0])

print("FINISH")

filename2 = "labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

print("[INFO] Loading file " + str(counter) + " ...")
start = time.time()
with open(filename2, 'r') as file:
    labels_temp = np.genfromtxt(file,dtype="string_")
length = labels_temp.shape[0] 
labels[0:length] = labels_temp
counter = counter + 1
end = time.time()
elapsed = end - start
print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
print("labels shape: " + str(labels.shape))
print("[INFO] length: " + str(length) )
while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    
    if os.path.isfile(filename) == False:
        break

    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file,dtype="string_")
    counter = counter + 1   
    
    new_len = labels_temp.shape[0] + length 
    labels[length:new_len] = labels_temp
    length = new_len
        
    end = time.time()
    elapsed = end - start
    print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
    print("labels shape: " + str(labels.shape))
    print("[INFO] length: " + str(length) )
    
print("labels shape: " + str(labels.shape))
print(labels)

