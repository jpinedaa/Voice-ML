import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.contrib import saved_model
from tensorflow.keras.layers import Dense, AveragePooling2D, Input, Conv2D, Concatenate, MaxPool2D, Reshape, BatchNormalization, Flatten, Lambda
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
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

def cosine_comparison(T, input_a, input_b):
    cosine_output = cosine_similarity(input_a, input_b)
    predicted_labels = []
    for i in range(len(input_a)):
        if cosine_output[i,i] < T:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    print("[INFO] input_a shape: " + str(input_a.shape) + " input_b shape: " + str(input_b.shape) + " predicted_labels shape: " + str(len(predicted_labels)))
    return predicted_labels

def euclidean_distance(T, input_a, input_b):
    x, y = input_a, input_b
    sum_square = np.sum(np.square(x - y), axis=1, keepdims=True)
    distance = np.sqrt(np.maximum(sum_square, 1e-7))
    print("distance shape: " + str(distance.shape))
    print(distance[5])
    predicted_labels = []
    for i in range(len(input_a)):
        if distance[i] < T:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
    return predicted_labels

def calculate_error(labels_true, labels_predicted):
    total_genuine = 0
    total_impostor = 0
    total_fr = 0
    total_fa = 0
    print(labels_true.shape)
    print(len(labels_predicted))
    for i in range(len(labels_true)):
        if labels_true[i] == 0:
            total_genuine += 1
            if labels_predicted[i] == 1:
                total_fr += 1
        else:
            total_impostor += 1
            if labels_predicted[i] == 0:
                total_fa += 1
    frr = total_fr/total_genuine
    far = total_fa/total_impostor
    acc = (total_fa + total_fr) / len(labels_predicted)
    return [frr,far, acc]

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpus', type=int, default=1, help='# of GPUs to use for training')
args = vars(ap.parse_args())
G = args["gpus"]

NUM_EPOCHS = 20
INIT_LR = 1e-5
lr_decay = 0
training_batch_size = 32
# samples_per_checkpoint = 1000
validation_split = 0.005
data_percent = 1
alpha = 1
logfile = "evaluation_log_6.txt"
graph_dir = "Graphs/"
update_name = "error_update3"
# checkpoint_path = "Saved_Models/training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# dir = "Saved_Model_4/"
save_dir = "Saved_Models/update4/"
save_dir2 = "Saved_Models/update4/"

print("[INFO] Searching Latest checkpoint... ")
checkpoints = [m for m in os.listdir(save_dir)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * margin_square + (1 - y_true) * square_pred)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))

# create the base pre-trained model
if G <= 1:
    print("[INFO] training with 1 GPU...")

    model = saved_model.load_keras_model(save_dir + checkpoints[-1])
    model = model.layers[-2]


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

rms = RMSprop()
model.compile(optimizer=SGD(INIT_LR,0.99), loss=contrastive_loss, metrics=[accuracy])
# model.compile(optimizer=RMSprop(lr=INIT_LR), loss='categorical_crossentropy',metrics=['accuracy'])


print("[INFO] Loading Data... ")
filename = "data4/data1.txt"
filename2 = "data4/labels1.txt"
counter = 1
le = preprocessing.LabelEncoder()

start = time.time()

filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]

data = np.memmap('pairs_data.array', dtype=np.float64, mode='r+', shape=(35000, 2, 100, 40, 3))
labels = np.memmap('pairs_labels.array', dtype= np.float64, mode= 'r+', shape= (35000, ))


print("[INFO] Finished Loading Data")
print("[INFO] Encoding Labels... ")

print(labels[0:200])

len_data = len(labels)
#le.fit(labels)
#labels = le.transform(labels)
# print(labels.shape)
#labels = to_categorical(labels, 2)
# labels = np.reshape(labels, (len_data,1,1,1251))

print("[INFO] Splitting Data to Training/Test splits ...")

test_size = 0.10
real_test_size = int(test_size * len_data)
x_train = np.memmap('x_train_pairs.array', dtype=np.float64, mode='r', shape=((len_data-real_test_size),2,100,40,3))
x_test = np.memmap('x_test_pairs.array', dtype=np.float64, mode='r', shape=(real_test_size,2,100,40,3))
y_train = np.memmap('y_train_pairs.array', dtype=np.float64, mode='r', shape=((len_data-real_test_size),))
y_test = np.memmap('y_test_pairs.array', dtype=np.float64, mode='r', shape=(real_test_size,))
"""
rng_state = np.random.get_state()
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(data[:len_data])
test_size = 0.10
real_test_size = int(test_size * len_data)
x_train1 = data[real_test_size:len_data]
x_test1 = data[:real_test_size]
y_train1 = labels[real_test_size:]
y_test1 = labels[:real_test_size]

x_train = np.memmap('x_train_pairs.array', dtype=np.float64, mode='w+', shape=x_train1.shape)
x_test = np.memmap('x_test_pairs.array', dtype=np.float64, mode='w+', shape=x_test1.shape)
y_train = np.memmap('y_train_pairs.array', dtype=np.float64, mode='w+', shape=y_train1.shape)
y_test = np.memmap('y_test_pairs.array', dtype=np.float64, mode='w+', shape=y_test1.shape)

x_train[:] = x_train1
x_test[:] = x_test1
y_train[:] = y_train1
y_test[:] = y_test1
"""
#x_train, x_test, y_train, y_test = train_test_split(data[0:len_data], labels , test_size=0.10, random_state= 42)
min = 100
error_rates = []
real_T = 0
i = 0
ind = 0
for T in np.linspace(-1,1,50):
    a = model.predict(x_test[:,0,:,:,:])
    b = model.predict(x_test[:, 1, :, :, :])
    predicted_labels = euclidean_distance(T, a, b)
    er = calculate_error(y_test,predicted_labels)
    temp = abs(er[0] - er[1])
    if  temp < min:
        min = temp
        real_T = T
        ind = i
    error_rates.append(er)
    i += 1




"""
print("[INFO] Plotting training loss and accuracy ...")
plt.plot(H['acc'])
plt.plot(H['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(graph_dir + update_name + "_acc")
plt.clf()
"""
far = [fa[1] for fa in error_rates]
frr = [fa[0] for fa in error_rates]
plt.plot(frr, far)
plt.title('error rate')
plt.ylabel('FAR')
plt.xlabel('FRR')
plt.savefig(graph_dir + update_name)




with open(logfile, 'a') as myfile:
    myfile.write(update_name + "real T: " + str(real_T) + " ratess: " + str(error_rates[ind]) + '\n')

print("FINISH")
