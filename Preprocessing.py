from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

rootdir = "/workspace/audio dataset/wav"

curr_id = "first"

myfile = open("data.txt", "w")
myfile = open("labels.txt", "w")

first_count = 0
count =0
start_time = 0
elapsed_time = 0
#no_samples = 500
for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
 #   if no_samples < 1:
 #       break
    if first_count == 0:
        first_count = dirs.__len__()
    if subdir.__len__() > 29:
        if subdir.__len__() < 38:
            if count!=0:
                elapsed_time = time.time() - start_time
            curr_id = subdir[29:38]
           # print(curr_id)
            count = count + 1
            start_time = time.time()

    for file in files:
        sound = AudioSegment.from_wav(subdir + "/" + file)
        sound = sound.set_channels(1)
        sound.export("modified.wav", format="wav")
        sample_rate, samples = wavfile.read("modified.wav")
        window = signal.get_window('hamming', int((sample_rate / 1000) * 25), True)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=window, noverlap= int((sample_rate / 1000) * 10))
        #print(frequencies.shape, times.shape)
        normalized_spec = (spectrogram - spectrogram.mean(axis=0)) / spectrogram.std(axis=0)
        #print(spectrogram.shape)
        #print(normalized_spec.shape)
        timevar = 300
        if normalized_spec.shape[1]>=timevar:
            no_cuts= int(normalized_spec.shape[1]/timevar)
            for i in range(no_cuts):
                cut= normalized_spec[:,i*timevar:(i*timevar)+timevar]
                #print("cut: ", cut.shape)
                with open("labels.txt", "a") as myfile:
                    myfile.write(curr_id + "\n")
                with open("data.txt", "a") as myfile:
                    np.savetxt(myfile, cut, delimiter= ',', newline="\n")
        else:
            pad = np.pad(normalized_spec,((0,0),(0,300 - normalized_spec.shape[1])), 'constant', constant_values=0)
            #print("pad: ", pad.shape)
            with open("labels.txt", "a") as myfile:
                myfile.write(curr_id + "\n")
            with open("data.txt", "a") as myfile:
                np.savetxt(myfile, pad, delimiter=',', newline="\n")
  #      no_samples = no_samples - 1
    if elapsed_time != 0:
        mins = int((elapsed_time * (first_count - count)) / 60)
        print("\rTotal Progress: ", int((count/first_count)*100), "% ------- estimated time left: ", int(mins/60) , "hours ", (mins%60), "mins ")
    else:
        print("\rTotal Progress: ", int((count / first_count) * 100), "% ------- estimated time left: ", "Calculating... ")

