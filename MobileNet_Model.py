import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
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

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpus', type=int, default=1, help= '# of GPUs to use for training')
args = vars(ap.parse_args())
G = args["gpus"]

NUM_EPOCHS = 100
INIT_LR= 0.001
logfile = "evaluation_log_2.txt"

# create the base pre-trained model
if G<= 1:
    print("[INFO] training with 1 GPU...")
    input= Input(shape=(201,300,1))
    in_conc = Concatenate()([input,input,input])
    base_model = MobileNet(weights='imagenet',input_tensor=in_conc ,include_top=False)
	x = base_model.output
    x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
    predictions = Dense(1251, activation='softmax')(x)
	model = Model(inputs=input, outputs=predictions)
else:
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        input= Input(shape=(201,300,1))
        in_conc = Concatenate()([input,input,input])
        base_model = MobileNet(weights='imagenet',input_tensor=in_conc ,include_top=False)
	    x = base_model.output
        x = GlobalAveragePooling2D()(x)
	    x = Dense(1024, activation='relu')(x)
        predictions = Dense(1251, activation='softmax')(x)
	    model = Model(inputs=input, outputs=predictions)
    model = multi_gpu_model(model, gpus=G)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

print("[INFO] Loading Data... ")
filename = "data1.txt"
filename2 = "labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

while 1:
    filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    counter = counter + 1
    if os.path.isfile(filename) == False:
        break
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
	
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.10)

    print("[INFO] Training starting ...")
    H = model.fit(x_train,y_train,verbose=1, epochs= NUM_EPOCHS, callbacks= callbacks)
    H = H.history
	
    print("[INFO] Plotting training loss and accuracy ...")
    N= np.arange(0, len(H["loss"]))
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(N, H['loss'], label= 'train_loss')
    plt.plot(N, H['acc'], label= 'train_acc')
    plt.title("Training Graph")
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(graph_dir + "training" + str(counter))
	
    print("[INFO] Saving Model ...")
    saved_model.save_keras_model(model,"Saved_Model_2")

    print("[INFO] Testing Model ...")
    R = model.evaluate(x_test, y_test, verbose=1)
    H = R.history
	
	with open(logfile, 'a') as myfile:
        myfile.write("batch number: " + str(counter) + '\n')
        resultstr = ' '.join(str(x) for x in R)
        myfile.write(resultstr + '\n')
	
    print("[INFO] Plotting testing loss and accuracy ...")
    N= np.arange(0, len(H["loss"]))
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(N, H['loss'], label= 'test_loss')
    plt.plot(N, H['acc'], label= 'test_acc')
    plt.title("Testing Graph")
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(graph_dir + "testing" + str(counter))
	
    counter = counter + 1



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
