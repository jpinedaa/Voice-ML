import numpy as np


def LoadData(file):
    data = np.genfromtxt(file, delimiter=",")
    #print("load data:", data.shape)
    samples = int(data.shape[0]/201)
    data = data.reshape(samples,201,300,1)
    #print("load data2:",data.shape)
    return data

