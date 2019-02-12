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
