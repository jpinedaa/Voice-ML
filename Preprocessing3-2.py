import numpy as np
import os
from LoadData import LoadData
import random

print("[INFO] Loading Data... ")
filename = "verdata/data1.txt"
filename2 = "verdata/labels1.txt"
counter = 1

filename = filename[:-4 - len(str(counter - 1))] + str(counter) + filename[-4:]
filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]

print("[INFO] Loading first file... ")

with open(filename2, 'r') as file:
    labels = np.genfromtxt(file, dtype="string_")
counter = counter + 1

with open(filename, "r") as file:
    data = LoadData(file)

counter = counter + 1

print("[INFO] Finished loading first file")
print("labels shape: " + str(labels.shape))

while 1:
    print("[INFO] Loading file " + str(counter) + " ...")

    filename = filename[:-4 - len(str(counter - 1))] + str(counter) + filename[-4:]
    filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]

    if os.path.isfile(filename2) == False:
        break
    with open(filename2, 'r') as file:
        labels_temp = np.genfromtxt(file, dtype="string_")
    with open(filename, "r") as file:
        data_temp = LoadData(file)
    counter = counter + 1

    labels = np.concatenate((labels, labels_temp))
    data = np.concatenate((data, data_temp))

    print("[INFO] Finished loading file")
    print("labels shape: " + str(labels.shape))

print("[INFO]Arrangig first pair... ")
pairs_data = np.zeros((1,2,100,40,3))
pairs_data[0] = [data[0],data[1]]
print("pairs data shape: " + str(pairs_data.shape))
pairs_labels = np.zeros((1,))
if labels[0] == labels[1]:
    pairs_labels[0] = 0
else:
    pairs_labels[0] = 1
flag = 1
i = 1
while i < (labels.shape[0] - 1):
    print("[INFO]Arranging pair#" + str(i))
    if flag == 0:
        pairs_data_tmp = np.zeros((1, 2, 100, 40, 3))
        pairs_data_tmp[0] = [data[i], data[i+1]]
        pairs_data = np.concatenate((pairs_data,pairs_data_tmp))
        pairs_labels_tmp = np.zeros((1,))
        if labels[i] == labels[i+1]:
            pairs_labels_tmp[0] = 0
        else:
            pairs_labels_tmp[0] = 1
        pairs_labels = np.concatenate((pairs_labels, pairs_labels_tmp))
        flag = 1
        i = i + 2
    if flag == 1:
        random.seed()
        index = random.randint(0,(labels.shape[0]-1))
        pairs_data_tmp = np.zeros((1, 2, 100, 40, 3))
        pairs_data_tmp[0] = [data[i], data[index]]
        pairs_data = np.concatenate((pairs_data, pairs_data_tmp))
        pairs_labels_tmp = np.zeros((1,))
        if labels[i] == labels[index]:
            pairs_labels_tmp[0] = 0
        else:
            pairs_labels_tmp[0] = 1
        pairs_labels = np.concatenate((pairs_labels, pairs_labels_tmp))
        flag = 1
        i = i + 1
    print("[INFO]pairs_data shape: " + str(pairs_data.shape) + " pairs_labels shape: " + str(pairs_labels.shape))

pairs_data1 = np.memmap('pairs_data.array', dtype= np.float64, mode= 'w+', shape= pairs_data.shape)
pairs_labels1 = np.memmap('pairs_labels.array', dtype= np.float64, mode= 'w+', shape= pairs_labels.shape)

pairs_data1[:] = pairs_data
pairs_labels1[:] = pairs_labels

print("final shapes= data:" + str(pairs_data.shape) + "labels: " + str(pairs_labels.shape))