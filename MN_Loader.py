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

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpus', type=int, default=1, help= '# of GPUs to use for training')
args = vars(ap.parse_args())
G = args["gpus"]

NUM_EPOCHS = 1000
INIT_LR= 0.000001
training_batch_size = 64
samples_per_checkpoint = 1000
validation_split = 0.10
logfile = "evaluation_log_3.txt"
graph_dir = "Graphs/minibatches64epoch1000/"
dir = "Saved_Model_3/"
save_dir = "Saved_Model_4/"

print("[INFO] Searching Latest checkpoint... ")
checkpoints = [m for m in os.listdir(dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

# create the base pre-trained model
if G<= 1:
    print("[INFO] training with 1 GPU...")
    model = saved_model.load_keras_model(dir + checkpoints[-1])
else:
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        model = saved_model.load_keras_model(dir + checkpoints[-1])
    model = multi_gpu_model(model, gpus=G)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
print("[INFO] Compiling Model ... ")
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

print("[INFO] Loading Data... ")
filename = "data1.txt"
filename2 = "labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

start = time.time()

filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 

data = np.memmap('data.array', dtype= np.float64, mode= 'w+', shape= (250000,201,300,1))
labels = np.empty((250000,),dtype= np.unicode_)

print("[INFO] Loading first file... ")

with open(filename, "r") as file:
    data_temp = LoadData(file)
with open(filename2, 'r') as file:
    labels_temp = np.genfromtxt(file,dtype="string_")

length = data_temp.shape[0] 
data[0:length] = data_temp
labels[0:labels_temp.shape[0]] = labels_temp
counter = counter + 1

end = time.time()
elapsed = end - start

print("[INFO] Finished loading first file, elapsed time: " + str(elapsed))
print("[INFO] data shape: " + str(data.shape) + "labels shape: " + str(labels.shape))
print("[INFO] length: " + str(length) )

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    
    if os.path.isfile(filename) == False:
        break
    with open(filename, "r") as file:
        data_temp = LoadData(file)
    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file,dtype="string_")
    counter = counter + 1   
    
    new_len = data_temp.shape[0] + length 
    data[length:new_len] = data_temp
    labels[length:new_len] = labels_temp
    length = new_len
        
    end = time.time()
    elapsed = end - start
    print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
    print("[INFO] data shape: " + str(data.shape) + "labels shape: " + str(labels.shape))
    print("[INFO] length: " + str(length) )
        
print("[INFO] Finished Loading Data")
print("[INFO] Encoding Labels... ")
        
le.fit(labels)
labels = le.transform(labels)
#print(labels.shape)
labels = to_categorical(labels, 1251)

print("[INFO] Splitting Data to Training/Test splits ...")
counter2 = 1
counter3 = 1
counter4 = 1
x_train, x_test, y_train, y_test = train_test_split(data[0:length], labels[0:length], test_size=0.10)
n_samples = len(y_train)
batch_size = int(training_batch_size/(1 - validation_split))
b_batch_size = -(-n_samples//-(-n_samples//batch_size))

print("[INFO] Training Starting... ")
for i in range(-(-n_samples//b_batch_size)):
    batch_x = x_train[i*b_batch_size:(i+1)*b_batch_size]
    print("[INFO] minibatch shape: " + str(batch_x.shape))
    batch_y = y_train[i*b_batch_size:(i+1)*b_batch_size]

    print("[INFO] Training minibatch#" + str(counter2))
    H = model.fit(batch_x,batch_y,verbose=1, epochs= NUM_EPOCHS, validation_split= validation_split)
    H = H.history

    print("[INFO] Plotting training loss and accuracy ...")
    N= np.arange(0, len(H["loss"]))
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(N, H['loss'], label= 'train_loss')
    plt.title("Training Graph")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(graph_dir + "training" + str(counter) + "_" + str(counter2))

    counter2 = counter2 + 1
    counter3 = counter3 + 1

    if counter3 == int(samples_per_checkpoint/b_batch_size):
        counter3 = 1
        print("[INFO] Saving Model ...")
        saved_model.save_keras_model(model, save_dir)

        print("[INFO] Testing Model ...")
        H = model.evaluate(x_test, y_test, verbose=1)

        with open(logfile, 'a') as myfile:
            myfile.write("checkpoint number: " + str(counter4) + '\n')
            myfile.write("loss: " + str(H[0]) + "accuracy: " + str(H[1]) + '\n')
            counter4 = counter4 + 1

    
print("FINISH")
