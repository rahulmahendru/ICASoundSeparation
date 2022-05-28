import numpy as np
np.random.seed(0)
from scipy.io import wavfile as wf
from util import *
from pydub import AudioSegment

# File Paths
files = ['male1.wav', 'female1.wav', 'male2.wav', 'female2.wav']
path = './audio/'

# Convert wav to data and mix the sounds
# Male + Female (6:45 duration)
frq1, sound1, _ = read_wav(path + files[0])
_, sound2, _ = read_wav(path + files[1])

# Male + Female (0:20 duration)
frq2, sound3, _ = read_wav(path + files[2])
_, sound4, _ = read_wav(path + files[3])

# Mix the sounds using X = As
mix1 = np.c_[center_data(sound1), center_data(sound2)]
mix2 = np.c_[center_data(sound3), center_data(sound4)]
A = np.array([[1, 0.5], [0.5, 1]])
signals1 = np.dot(mix1, A).T 
signals2 = np.dot(mix2, A).T 

# # Convert array into a matrix of stacked rows
d1 = np.vstack(signals1)
d2 = np.vstack(signals2)

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
X1 = whiten_matrix(center_data(d1))
X2 = whiten_matrix(center_data(d2))

# Find individual components
tolerance = 0.000001
iterations = 100

#  Implement the FastICA
S1 = fastica(X1, iterations, tolerance)
S2 = fastica(X2, iterations, tolerance)

#  Write the files
wf.write('t11_predicted.wav', frq1, S1[0].astype(np.float32))
wf.write('t21_predicted.wav', frq1, S1[1].astype(np.float32))
wf.write('t12_predicted.wav', frq2, S2[0].astype(np.float32))
wf.write('t22_predicted.wav', frq2, S2[1].astype(np.float32))

# Calculate the Mean Square Error
MSE1 = np.square(np.subtract(sound1, S1[0]+ center_data(sound1))).mean()
MSE2 = np.square(np.subtract(sound2, S1[1]+ center_data(sound2))).mean()
MSE3 = np.square(np.subtract(sound4, S2[0]+ center_data(sound4))).mean()
MSE4 = np.square(np.subtract(sound3, S2[1]+ center_data(sound3))).mean()

# Print the Mean Square Error Values
print("The MSE1 value is: ", MSE1)
print("The MSE2 value is: ", MSE2)
print("The MSE3 value is: ", MSE3)
print("The MSE4 value is: ", MSE4)



