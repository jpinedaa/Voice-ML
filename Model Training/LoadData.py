import numpy as np


def LoadData(file):
    data = np.genfromtxt(file, delimiter=",")
    #print("load data:", data.shape)
    print(data.shape)
    samples = int(int(data.shape[0]/40)/100)
    data = data.reshape(samples,100,40,3)
    #print("load data2:",data.shape)
    return data

