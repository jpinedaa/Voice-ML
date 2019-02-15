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
filename = "data2/data.txt"
filename2 = "data2/labels.txt"

first_count = 0
count =0
start_time = 0
elapsed_time = 0
batch_size = 5000
batch_ptr = batch_size
counter = 1

filename = filename[:-4] + str(counter) + filename[-4:] 
filename2 = filename2[:-4] + str(counter) + filename2[-4:] 
for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    if batch_ptr < 1:
        batch_ptr=batch_size
        counter = counter + 1
        filename = filename[:-4-len(str(counter-1))] + str(counter) + filename[-4:] 
        filename2 = filename2[:-4-len(str(counter-1))] + str(counter) + filename2[-4:] 
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
        window = signal.get_window('hamming', 1024, True)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, window=window, noverlap= 512)
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
                with open(filename2, "a") as myfile:
                    myfile.write(curr_id + "\n")
                with open(filename, "a") as myfile:
                    np.savetxt(myfile, cut, delimiter= ',', newline="\n")
        else:
            pad = np.pad(normalized_spec,((0,0),(0,300 - normalized_spec.shape[1])), 'constant', constant_values=0)
            #print("pad: ", pad.shape)
            with open(filename2, "a") as myfile:
                myfile.write(curr_id + "\n")
            with open(filename, "a") as myfile:
                np.savetxt(myfile, pad, delimiter=',', newline="\n")
        batch_ptr = batch_ptr - 1
    if elapsed_time != 0:
        mins = int((elapsed_time * (first_count - count)) / 60)
        print("\rTotal Progress: ", int((count/first_count)*100), "% ------- estimated time left: ", int(mins/60) , "hours ", (mins%60), "mins ")
    else:
        print("\rTotal Progress: ", int((count / first_count) * 100), "% ------- estimated time left: ", "Calculating... ")

