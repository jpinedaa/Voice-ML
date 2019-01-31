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
checkpoints = [m for m in os.listdir(dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

model = saved_model.load_keras_model(dir + checkpoints[-1])
model.summary
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss= 'categorical_crossentropy', metrics=['accuracy'])

filename = "data0.txt"
filename2 = "labels0.txt"
counter = 1
le = preprocessing.LabelEncoder()

while 1:
    filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    counter = counter + 1
    if os.path.isfile(filename) == False:
        break
    print("loading data")
    with open(filename, "r") as file:
        data = LoadData(file)
    with open(filename2, 'r') as file:
        labels = np.genfromtxt(file,dtype="string_")
    print("encoding labels")
    le.fit(labels)
    labels = le.transform(labels)
    #print(labels.shape)
    print("converting labels to categorical matrix")
    labels = to_categorical(labels, 1251)
	
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.05)

    model.fit(x_train,y_train,verbose=1, epochs= 100)

    saved_model.save_keras_model(model,"Saved_Model")

    print(model.evaluate(x_test, y_test, verbose=1))
	
	
	
print("FINISH")
