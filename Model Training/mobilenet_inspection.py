import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, MaxPool2D , Input, Conv2D, Concatenate, Reshape, BatchNormalization, Flatten,Lambda
from tensorflow.keras.activations import softmax
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
from tensorflow.keras import backend as K


NUM_EPOCHS = 300
INIT_LR = 0.00001
alpha = 1
batch_size = 128
logfile = "evaluation_log_2.txt"
graph_dir = "Graphs/minibatches128lr0.00001/"

# create the base pre-trained model

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

input = Input(shape=(100, 40, 3))
base_model = MobileNet(input_shape=(100, 40, 3), weights=None, input_tensor=input, include_top=False)
x = base_model.output
#x = MaxPool2D(pool_size=(2, 2))(x)
# model = Model(inputs=input, outputs= x)
# layer = model.get_layer(index = -1)
# print(layer.output_shape)
#x = Conv2D(4096, kernel_size=(8, 1), activation='relu')(x)
# x = AveragePooling2D(pool_size=(1,4))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1251, activation='softmax')(x)
x = Reshape(target_shape=(1251,))(x)
model = Model(inputs=input, outputs=x)
model.layers.pop()
x = model.output
x = Dense(1024, activation='relu', name='features')(x)
model2 = Model(model.input, x)

input_a = Input(shape=(100, 40, 3))
input_b = Input(shape=(100, 40, 3))

processed_a = model2(input_a)
processed_b = model2(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model3 = Model([input_a, input_b], distance)

print(model3.summary())

for i in range(len(model3.layers)):
    layer = model3.get_layer(index = -i)
    print(layer.output_shape)

temp_weights = [layer.get_weights() for layer in model3.layers]
"""print((temp_weights[-4]))
print((temp_weights[-3]))
print((temp_weights[-2]))
print((temp_weights[-1]))"""


input = Input(shape=(100, 40, 3))
base_model = MobileNet(input_shape=(100, 40, 3), weights=None, input_tensor=input, include_top=False)
x = base_model.output
#x = MaxPool2D(pool_size=(2, 2))(x)
# model = Model(inputs=input, outputs= x)
# layer = model.get_layer(index = -1)
# print(layer.output_shape)
#x = Conv2D(4096, kernel_size=(8, 1), activation='relu')(x)
# x = AveragePooling2D(pool_size=(1,4))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1251, activation='softmax')(x)
x = Reshape(target_shape=(1251,))(x)
model = Model(inputs=input, outputs=x)
model.layers.pop()
x = model.output
x = Dense(1024, activation='relu', name='features')(x)
model = Model(inputs=input, outputs=x)

print(model.summary())
print(len(model.layers))
print(len(temp_weights[-2]))

for i in range(len(temp_weights[-2])):
    model.layers[i].set_weights(temp_weights[-2][i])

print("FINISH")

"""
temp_weights = [layer.get_weights() for layer in model.layers]
inp = Input(shape=(100,40,3))
inp2 = BatchNormalization()(inp)
base_model = MobileNet(input_shape=(100,40,3), weights = None, input_tensor= inp2, include_top=False)
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation= 'relu')(x)
x = Dense(1251, activation='softmax')(x)
x = Reshape(target_shape=(1251,))(x)
model = Model(inputs=inp, outputs=x)
j = 0
for i in range(len(temp_weights)):
    print("i: " + str(i) + " j: " + str(j) )
    if j == 1 :
        j = j + 1
    model.layers[j].set_weights(temp_weights[i])
    j = j + 1"""