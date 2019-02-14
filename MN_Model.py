import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, Concatenate
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.train import latest_checkpoint
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


NUM_EPOCHS = 200
INIT_LR= 0.001
lr_decay = 2
training_batch_size = 32
#samples_per_checkpoint = 1000
validation_split = 0.05
data_percent = 0.10
alpha = 1
logfile = "evaluation_log_4.txt"
graph_dir = "Graphs/"
graph_name = "512batchsize"
checkpoint_path = "Saved_Models/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#dir = "Saved_Model_4/"
save_dir = "Saved_Model_5/"


def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = lr_decay
    alpha = baseLR * (1 - (epoch/float(maxEpochs)))**power
    return alpha

# create the base pre-trained model
if G<= 1:
    print("[INFO] training with 1 GPU...")
    input= Input(shape=(201,300,1))
    in_conc = Concatenate()([input,input,input])
    base_model = MobileNet(weights='imagenet',input_tensor=in_conc ,include_top=True, alpha= alpha)
    x = base_model.output
    predictions = Dense(1251, activation='softmax')(x)
    model = Model(inputs=input, outputs=predictions)
else:
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        input= Input(shape=(201,300,1))
        in_conc = Concatenate()([input,input,input])
        base_model = MobileNet(weights='imagenet',input_tensor=in_conc ,include_top=True, alpha= alpha)
        x = base_model.output
        predictions = Dense(1251, activation='softmax')(x)
        model = Model(inputs=input, outputs=predictions)
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

filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 

data = np.memmap('data.array', dtype= np.float64, mode= 'r+', shape= (250000,201,300,1))

print("[INFO] Loading first file... ")

with open(filename2, 'r') as file:
    labels = np.genfromtxt(file,dtype="string_")
counter = counter + 1

end = time.time()
elapsed = end - start

print("[INFO] Finished loading first file, elapsed time: " + str(elapsed))
print("labels shape: " + str(labels.shape))

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
    
    if os.path.isfile(filename2) == False:
        break
    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file,dtype="string_")
    counter = counter + 1   
    
    labels = np.concatenate((labels, labels_temp))
        
    end = time.time()
    elapsed = end - start
    print("[INFO] Finished loading file, elapsed time: " + str(elapsed))
    print("labels shape: " + str(labels.shape))
        
print("[INFO] Finished Loading Data")
print("[INFO] Encoding Labels... ")
        
len_data = len(labels)
le.fit(labels)
labels = le.transform(labels)
#print(labels.shape)
labels = to_categorical(labels, 1251)

print("[INFO] Splitting Data to Training/Test splits ...")
x_train, x_test, y_train, y_test = train_test_split(data[0:len_data], labels, test_size=0.20, random_state= 42)
with open(logfile, 'a') as myfile:
    myfile.write("x_train shape: " + str(x_train.shape) + "y_train shape: " + str(y_train.shape) + '\n')
    
print("[INFO] Training with " + str(data_percent*100) + "% of training data")
newlen = int(len(y_train)*data_percent)
x_train_new = x_train[0:newlen]
y_train_new = y_train[0:newlen]

print("x_train_new shape: " + str(x_train_new.shape) + "y_train_new shape: " + str(y_train_new.shape))
    
#Callback functions
cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only = True)
lr_callback = LearningRateScheduler(poly_decay, verbose=1)

print("[INFO] Training starting... ")
H = model.fit(x_train_new,y_train_new,batch_size= training_batch_size ,verbose=1, epochs= NUM_EPOCHS, validation_split= validation_split, callbacks= [cp_callback, lr_callback])
H = H.history

print("[INFO] Plotting training loss and accuracy ...")
plt.plot(H['acc'])
plt.plot(H['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(graph_dir + graph_name + "_acc")

plt.plot(H['loss'])
plt.plot(H['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(graph_dir + graph_name + "_loss")

print("[INFO] Saving Model ...")
saved_model.save_keras_model(model, save_dir)

print("[INFO] Testing Model ...")
H = model.evaluate(x_test, y_test, verbose=1)

with open(logfile, 'a') as myfile:
    myfile.write("loss: " + str(H[0]) + "accuracy: " + str(H[1]) + '\n')

    
print("FINISH")
