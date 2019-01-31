import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from LoadData import LoadData
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os


dir = "Saved_Model/"
logfile = "evaluation_log.txt"

checkpoints = [m for m in os.listdir(dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

with open("data30.txt", "r") as file:
    data = LoadData(file)
with open("labels30.txt", 'r') as file:
    labels = np.genfromtxt(file,dtype="string_")
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
#print(labels.shape)
labels = to_categorical(labels, 1251)	

counter= 1		
for checkpoint in checkpoints:
    model = saved_model.load_keras_model(dir + checkpoint)
    model.compile(optimizer = SGD(lr=0.0001, momentum= 0.9), loss= 'categorical_crossentropy', metrics= ['accuracy'])
    with open(logfile, 'a') as myfile:
        myfile.write(str(counter) + "model number: " + checkpoint + '\n')
        result = model.evaluate(data, labels, verbose=1)
        resultstr = ' '.join(str(x) for x in result)
        myfile.write(resultstr + '\n')
	
    counter = counter + 1
