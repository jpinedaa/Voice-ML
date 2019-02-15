import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
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

print("[INFO] Loading Data... ")
filename = "data1.txt"
filename2 = "labels1.txt"
counter = 1

start = time.time()

filename = filename[:-4 - len(str(counter - 1))] + str(counter) + filename[-4:]

data = np.memmap('data2.array', dtype=np.float64, mode='w+', shape=(250000, 201, 300, 1))

print("[INFO] Loading first file... ")

with open(filename, "r") as file:
    data_temp = LoadData(file)

length = data_temp.shape[0]
data[0:length] = data_temp
counter = counter + 1

end = time.time()
elapsed = end - start

print("[INFO] Finished loading first file, elapsed time: " + str(elapsed))
print("[INFO] data shape: " + str(data.shape))
print("[INFO] length: " + str(length))

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename = filename[:-4 - len(str(counter - 1))] + str(counter) + filename[-4:]

    if os.path.isfile(filename) == False:
        break
    with open(filename, "r") as file:
        data_temp = LoadData(file)
    counter = counter + 1

    new_len = data_temp.shape[0] + length
    data[length:new_len] = data_temp
    length = new_len

    end = time.time()
    elapsed = end - start
    print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
    print("[INFO] data shape: " + str(data.shape))
    print("[INFO] length: " + str(length))

print("[INFO] Finished Loading Data")

print("FINISH")
