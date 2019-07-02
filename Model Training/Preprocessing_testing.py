from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from speechpy.feature import lmfe, extract_derivative_feature
import os
import time
import librosa
from feature import lmfe, extract_derivative_feature

rootdir = "D:\documents/audio dataset/vox1_dev_wav/wav"
file = "\id10001/1zcIwhmdeo4/00001.wav"
#rootdir = ""
#file = "D:\downloads/440Hz_44100Hz_16bit_30sec.wav"

sound = AudioSegment.from_wav(rootdir + file)
sound = sound.set_channels(1)
sound.export("modified.wav", format="wav")
sample_rate, samples = wavfile.read("modified.wav")
print("sample rate =" + str(sample_rate))
print(samples.shape)
print(samples[500])
features = lmfe(samples,sample_rate,0.025, 0.01, 40, 512)
print(features.shape)
print(features[1][1])
features = extract_derivative_feature(features)
print(features.shape)
print(features)
#ax = sns.heatmap(features[:,:,0])
#//plt.show()
#//plt.clf()
#//ax = sns.heatmap(features[:,:,1])
#//plt.show()
#//plt.clf()
#ax = sns.heatmap(features[:,:,2])
#plt.show()
#plt.clf()

"""window = signal.get_window('hamming', 1024, True)
frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate, window=window, noverlap= 512)
#spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
print("frequencies shape: "+ str(frequencies.shape) + " times shape: " + str(times.shape) + " spectrogram shape: " + str(spectrogram.shape))
print(frequencies)
print(times)
ax = sns.heatmap(spectrogram)
plt.show()
normalized_spec = spectrogram
for i in range(spectrogram.shape[1]):
    normalized_spec[:,i:i+1] = normalized_spec[:,i:i+1] - spectrogram.mean(axis=1).reshape((513,1))
    normalized_spec[:,i:i+1] = normalized_spec[:,i:i+1] / spectrogram.std(axis=1).reshape(513,1)
print("spec: " + str(spectrogram.shape) + "spec mean: " + str(spectrogram.mean(axis=1).shape) + "spec var: " + str(spectrogram.std(axis=1).shape))
ax = sns.heatmap(normalized_spec)
plt.show()
"""
"""
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
"""