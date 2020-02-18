# Temporary code to load and evaluate the data 
# Using the FMA small dataset
# Code to read sample data and provide a simple plot and spectogram

import librosa
#%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display

print('Hello\n')

audio_path = '..\\Dataset\\fma_small\\000\\000002.mp3'
x , sr = librosa.load(audio_path)
print(type(x),type(sr))

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.show()