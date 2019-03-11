import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Conv2D, Concatenate, MaxPool2D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.optimizers import SGD, RMSprop
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
from pandas import DataFrame

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpus', type=int, default=1, help='# of GPUs to use for training')
args = vars(ap.parse_args())
G = args["gpus"]

NUM_EPOCHS = 100
INIT_LR = 1e-5
lr_decay = 0
training_batch_size = 32
# samples_per_checkpoint = 1000
validation_split = 0.005
data_percent = 1
alpha = 1
logfile = "evaluation_log_5.txt"
graph_dir = "Graphs/"
update_name = "update18"
# checkpoint_path = "Saved_Models/training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# dir = "Saved_Model_4/"
save_dir = "Saved_Models/update3/"

print("[INFO] Searching Latest checkpoint... ")
checkpoints = [m for m in os.listdir(save_dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = lr_decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


# create the base pre-trained model
if G <= 1:
    print("[INFO] training with 1 GPU...")
    """
    input = Input(shape=(100, 40, 3))
    base_model = MobileNet(input_shape=(100, 40, 3), weights=None, input_tensor=input, include_top=False)
    x = base_model.output
    x = MaxPool2D(pool_size=(2, 2))(x)
    # model = Model(inputs=input, outputs= x)
    # layer = model.get_layer(index = -1)
    # print(layer.output_shape)
    x = Conv2D(4096, kernel_size=(8, 1), activation='relu')(x)
    # x = AveragePooling2D(pool_size=(1,4))(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1251, activation='softmax')(x)
    x = Reshape(target_shape=(1251,))(x)
    model = Model(inputs=input, outputs=x)"""

    model = saved_model.load_keras_model(save_dir + checkpoints[-1])

    temp_weights = [layer.get_weights() for layer in model.layers]
    inp = Input(shape=(100,40,3))
    inp2 = BatchNormalization()(inp)
    base_model = MobileNet(input_shape=(100,40,3), weights = None, input_tensor= inp2, include_top=False)
    x = base_model.output
    x = Conv2D(4096, kernel_size=(4, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(1,2))(x)
    x = Dense(1024, activation='relu')(x)
    x = Reshape(target_shape=(1024,))(x)
    x = BatchNormalization()(x)
    x = Dense(1251, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    for i in range(97):
        model.layers[i].set_weights(temp_weights[i])

   """ j = 0
    for i in range(len(temp_weights)):
        print("i: " + str(i) + " j: " + str(j) )
        if j == 1 :
            j = j + 1
        model.layers[j].set_weights(temp_weights[i])
        j = j + 1"""

else:
    print("[INFO] training with {} GPUs...".format(G))
    with tf.device("/cpu:0"):
        """
        input = Input(shape=(100, 40, 3))
        base_model = Xception(input_shape=(100, 40, 3), weights=None, input_tensor=input, include_top=False)
        x = base_model.output
        x = MaxPool2D(pool_size=(2, 2))(x)
        # model = Model(inputs=input, outputs= x)
        # layer = model.get_layer(index = -1)
        # print(layer.output_shape)
        x = Conv2D(4096, kernel_size=(8, 1), activation='relu')(x)
        # x = AveragePooling2D(pool_size=(1,4))(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1251, activation='softmax')(x)
        x = Reshape(target_shape=(1251,))(x)
        model = Model(inputs=input, outputs=x)
        """
        model = saved_model.load_keras_model(save_dir + checkpoints[-1])
    model = multi_gpu_model(model, gpus=G)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
print("[INFO] Compiling Model ... ")
from tensorflow.keras.optimizers import SGD

model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(lr=INIT_LR), loss='categorical_crossentropy',metrics=['accuracy'])


print("[INFO] Loading Data... ")
filename = "data4/data1.txt"
filename2 = "data4/labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

start = time.time()

filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]

data = np.memmap('data4.array', dtype=np.float64, mode='r+', shape=(320000, 100, 40, 3))

print("[INFO] Loading first file... ")

with open(filename2, 'r') as file:
    labels = np.genfromtxt(file, dtype="string_")
counter = counter + 1

end = time.time()
elapsed = end - start

print("[INFO] Finished loading first file, elapsed time: " + str(elapsed))
print("labels shape: " + str(labels.shape))

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")
    start = time.time()

    filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]

    if os.path.isfile(filename2) == False:
        break
    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file, dtype="string_")
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
# print(labels.shape)
labels = to_categorical(labels, 1251)
# labels = np.reshape(labels, (len_data,1,1,1251))

print("[INFO] Splitting Data to Training/Test splits ...")
test_size = 0.20
real_test_size = int(0.20 * len_data)
x_train = np.memmap('x_train2.array', dtype=np.float64, mode='r', shape=((len_data-real_test_size),100,40,3))
x_test = np.memmap('x_test2.array', dtype=np.float64, mode='r', shape=(real_test_size,100,40,3))
y_train = np.memmap('y_train2.array', dtype=np.float64, mode='r', shape=((len_data-real_test_size),1251,))
y_test = np.memmap('y_test2.array', dtype=np.float64, mode='r', shape=(real_test_size,1251,))
"""
rng_state = np.random.get_state()
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(data[:len_data])
test_size = 0.20
real_test_size = int(0.20 * len_data)
x_train1 = data[real_test_size:len_data]
x_test1 = data[:real_test_size]
y_train1 = labels[real_test_size:]
y_test1 = labels[:real_test_size]

x_train = np.memmap('x_train2.array', dtype=np.float64, mode='w+', shape=x_train1.shape)
x_test = np.memmap('x_test2.array', dtype=np.float64, mode='w+', shape=x_test1.shape)
y_train = np.memmap('y_train2.array', dtype=np.float64, mode='w+', shape=y_train1.shape)
y_test = np.memmap('y_test2.array', dtype=np.float64, mode='w+', shape=y_test1.shape)

x_train[:] = x_train1
x_test[:] = x_test1
y_train[:] = y_train1
y_test[:] = y_test1
"""
# x_train, x_test, y_train, y_test = train_test_split(data[0:len_data], labels , test_size=0.20, random_state= 42)
with open(logfile, 'a') as myfile:
    myfile.write("x_train shape: " + str(x_train.shape) + "y_train shape: " + str(y_train.shape) + '\n')

print("[INFO] Training with " + str(data_percent * 100) + "% of training data")
newlen = int(len(y_train) * data_percent)
x_train_new = x_train[0:newlen]
y_train_new = y_train[0:newlen]

print("x_train_new shape: " + str(x_train_new.shape) + "y_train_new shape: " + str(y_train_new.shape))

# Callback functions
# cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only = True)
lr_callback = LearningRateScheduler(poly_decay, verbose=1)

print("[INFO] Training starting... ")
H = model.fit(x_train_new, y_train_new, batch_size=training_batch_size, verbose=1, epochs=NUM_EPOCHS,
              validation_split=validation_split, callbacks=[lr_callback])
H = H.history

print("[INFO] Plotting training loss and accuracy ...")
plt.plot(H['acc'])
plt.plot(H['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(graph_dir + update_name + "_acc")
plt.clf()

plt.plot(H['loss'])
plt.plot(H['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(graph_dir + update_name + "_loss")

print("[INFO] Saving Model ...")
saved_model.save_keras_model(model, save_dir)

print("[INFO] Testing Model ...")
H = model.evaluate(x_test, y_test, verbose=1)

with open(logfile, 'a') as myfile:
    myfile.write(update_name + " epochs= " + str(NUM_EPOCHS) + " lr= " + str(INIT_LR) + " loss: " + str(
        H[0]) + " accuracy: " + str(H[1]) + '\n')

print("FINISH")
