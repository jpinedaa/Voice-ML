import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.contrib import saved_model
from LoadData import LoadData
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os


# create the base pre-trained model
input= Input(shape=(201,300,1))
in_conc = Concatenate()([input,input,input])
base_model = MobileNet(weights=None,input_tensor=in_conc ,include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

predictions = Dense(1251, activation='softmax')(x)

# this is the model we will train

model = Model(inputs=input, outputs=predictions)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

saved_model.save_keras_model(model,  'Saved_Model')

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

    saved_model.save_keras_model(model,  'Saved_Model')

    print(model.evaluate(x_test, y_test, verbose=1))

"""	
print("loading data")
with open("data.txt", "r") as file:
    data = LoadData(file)
with open("labels.txt", 'r') as file:
    labels = np.genfromtxt(file,dtype="string_")

#print(data.shape)
#print(labels.shape)
#print(labels)
#print(data)
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

print("encoding labels")
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
#print(labels.shape)
print("Num of Speakers: ", le.get_params().size)
print("converting labels to categorical matrix")
labels = to_categorical(labels, 1251)
#print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.20)

model.fit(x_train,y_train,verbose=1, epochs= 30)

model.save('MobileNet1') 

print(model.evaluate(x_test, y_test, verbose=1))"""


print("FINISH")
