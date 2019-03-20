from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
from speechpy.feature import lmfe, extract_derivative_feature

rootdir= "/workspace/audio dataset2/LibriSpeech/train-clean-360/"

curr_id = 0
filename = "verdata/data.txt"
filename2 = "verdata/labels.txt"

first_count = 0
count = 0
start_time = 0
elapsed_time = 0
batch_size = 5000
batch_ptr = batch_size
counter = 1
dirs_count=0
flag = 0

exit = 500

filename = filename[:-4] + str(counter) + filename[-4:]
filename2 = filename2[:-4] + str(counter) + filename2[-4:]
for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    if batch_ptr < 1:
        batch_ptr = batch_size
        counter = counter + 1
        filename = filename[:-4 - len(str(counter - 1))] + str(counter) + filename[-4:]
        filename2 = filename2[:-4 - len(str(counter - 1))] + str(counter) + filename2[-4:]
    if first_count == 0:
        first_count = dirs.__len__()
        flag = 1
        continue
    if flag == 1:
        curr_id = curr_id + 1
        dirs_count = dirs.__len__()
        print("dirs_count: " + str(dirs_count))
        flag = 0
        elapsed_time = time.time() - start_time
        start_time = time.time()
        count = count + 1
        continue
    if flag == 0:
        if dirs_count == 1:
            flag = 1
        dirs_count = dirs_count - 1

    """if subdir.__len__() > 29:
        if subdir.__len__() < 38:
            if count != 0:
                elapsed_time = time.time() - start_time
            curr_id = subdir[29:38]
            # print(curr_id)
            count = count + 1
            start_time = time.time()
"""
    no_files = 2
    for file in files:
        #exit = exit - 1
        if exit <= 0:
            break
        if file[-4:] != "flac":
            continue
        if no_files == 0:
            continue
        no_files = no_files - 1
        #print(subdir + "\\"  + file)
        sound = AudioSegment.from_file(subdir + "/" + file)
        sound = sound.set_channels(1)
        sound.export("modified.wav", format="wav")
        sample_rate, samples = wavfile.read("modified.wav")

        features = lmfe(samples, sample_rate, 0.025, 0.01, 40)
        features = extract_derivative_feature(features)

        timevar = 100
        if features.shape[0] >= timevar:
            no_cuts = int(features.shape[0] / timevar)
            for i in range(no_cuts):
                cut = features[i * timevar:(i * timevar) + timevar:,:,:]
                # print("cut: ", cut.shape)
                with open(filename2, "a") as myfile:
                    myfile.write(str(curr_id) + "\n")
                with open(filename, "a") as myfile:
                    for data_slice in cut:
                        np.savetxt(myfile, data_slice, delimiter=',', newline="\n")
        else:
            pad = np.pad(features, ((0, 100 - features.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
            # print("pad: ", pad.shape)
            with open(filename2, "a") as myfile:
                myfile.write(str(curr_id) + "\n")
            with open(filename, "a") as myfile:
                for data_slice in pad:
                    np.savetxt(myfile, data_slice, delimiter=',', newline="\n")
        batch_ptr = batch_ptr - 1
    if exit <= 0 :
        break
    if elapsed_time != 0:
        mins = int((elapsed_time * (first_count - count)) / 60)
        print("\rTotal Progress: ", int((count / first_count) * 100), "% ------- estimated time left: ", int(mins / 60),
              "hours ", (mins % 60), "mins ")
    else:
        print("\rTotal Progress: ", int((count / first_count) * 100), "% ------- estimated time left: ",
              "Calculating... ")
