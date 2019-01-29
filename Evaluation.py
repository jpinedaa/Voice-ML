from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

dir = "Saved_Model"
logfile = "evaluation_log.txt"

checkpoints = [m for m in os.listdir(dir) if os.path.isdir(dir + m)]
checkpoints = [int(x) for x in checkpoints]
checkpoints.sort()
checkpoints = [str(x) for x in checkpoints]

with open("data30.txt, "r") as file:
    data = LoadData(file)
with open("labels30,txt", 'r') as file:
    labels = np.genfromtxt(file,dtype="string_")
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
#print(labels.shape)
labels = to_categorical(labels, 1251)	
		
for checkpoint in checkpoints
    model = saved_model.load_keras_model(checkpoint)
    with open(logfile, 'a') as myfile
	    myfile.write("model number: " + checkpoint)
        myfile.write(model.summary)
		myfile.write(model.evaluate(data, labels, verbose=1))
	