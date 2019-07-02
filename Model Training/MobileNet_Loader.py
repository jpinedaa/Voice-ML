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

NUM_EPOCHS = 200
INIT_LR= 0.001

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - (epoch/float(maxEpochs)))**power
    return alpha

	
print("[INFO] Searching Latest checkpoint... ")
dir = "Saved_Model/"
checkpoints = [m for m in os.listdir(dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

if G<= 1:
    print("[INFO] training with 1 GPU...")
    model = saved_model.load_keras_model(dir + "1548336946")
else:
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        model = saved_model.load_keras_model(dir + "1548336946")
    model = multi_gpu_model(model, gpus=G)
	
print("[INFO] compiling model...")
model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9), loss= 'categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Loading Data... ")
filename = "data0.txt"
filename2 = "labels0.txt"
counter = 1
le = preprocessing.LabelEncoder()

graph_dir = 'Graphs/'

while 1:
    filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    
    if os.path.isfile(filename) == False:
        break
    with open(filename, "r") as file:
        data = LoadData(file)
    with open(filename2, 'r') as file:
        labels = np.genfromtxt(file,dtype="string_")
 
    le.fit(labels)
    labels = le.transform(labels)
    #print(labels.shape)
   
    labels = to_categorical(labels, 1251)
	
    callbacks = [LearningRateScheduler(poly_decay)]
	
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.10)

    print("[INFO] Training starting ...")
    H = model.fit(x_train,y_train,verbose=2, epochs= NUM_EPOCHS, callbacks= callbacks)
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
    H = model.evaluate(x_test, y_test, verbose=1)
    H = H.history
	
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
	
	
	
print("FINISH")
