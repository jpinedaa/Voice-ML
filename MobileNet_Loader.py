import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate
from tensorflow.keras.utils import to_categorical
from LoadData import LoadData
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os


model = keras.models.load_model("MobileNet1")
model.summary

filename = "data1.txt"
filename2 = "labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

while 1:
    filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    counter = counter + 1
    if os.path.isfile(filename) == False or counter>25:
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

    model.fit(x_train,y_train,verbose=1, epochs= 30)

    model.save('MobileNet2')

    print(model.evaluate(x_test, y_test, verbose=1))
	
	
	
print("FINISH")