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

print("shape: " + str(data.shape) )

print("data[0]:  ", data[0])

print("FINISH")

filename2 = "labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

print("[INFO] Loading file " + str(counter) + " ...")
start = time.time()
with open(filename2, 'r') as file:
    labels = np.genfromtxt(file,dtype="string_")
counter = counter + 1
end = time.time()
elapsed = end - start
print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
print("labels shape: " + str(labels.shape))

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    
    if os.path.isfile(filename2) == False:
        break

    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file,dtype="string_")
    counter = counter + 1   
    
    labels = np.concatenate((labels,labels_temp))
        
    end = time.time()
    elapsed = end - start
    print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
    print("labels shape: " + str(labels.shape))
    
print("labels shape: " + str(labels.shape))
print(labels[0:20])

le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
first, second = train_test_split(labels, test_size=0.10)
print(first[0:20])
print(second[0:20])
print(labels[0:20])
labels = to_categorical(labels, 1251)
print("shape: " + str(labels.shape))
print(labels[0:20])
