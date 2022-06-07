import numpy as np
np.random.seed(0)
from scipy.io import wavfile as wf
from util import *
import sys

# File Paths
files = ['male1.wav', 'female1.wav', 'male2.wav', 'female2.wav']
path = './audio/'

# Convert wav to data and mix the sounds
# Male + Female (6:45 duration)
frq1, sound1, time1 = read_wav(path + files[0])
_, sound2, _ = read_wav(path + files[1])

# Male + Female (0:20 duration)
frq2, sound3, time2 = read_wav(path + files[2])
_, sound4, _ = read_wav(path + files[3])

# Mix the sounds using X = As
mix1 = np.c_[center_data(sound1), center_data(sound2)]
mix2 = np.c_[center_data(sound3), center_data(sound4)]
A = np.array([[1, 0.5], [0.5, 1]])
signals1 = np.dot(mix1, A).T 
signals2 = np.dot(mix2, A).T 

# Plot the mixed data
plot_wav_test(signals1, 'Male + Female (6:45)')
plot_wav_test(signals2, 'Male + Female (0:20)')

# Scale the data for NMF
D1 = scale_data(signals1)
D2 = scale_data(signals2)

# Convert array into a matrix of stacked rows
d1 = np.vstack(signals1)
d2 = np.vstack(signals2)

# Store the matrices as weights 
np.save('./weights/d1_test.npy', d1)
np.save('./weights/d2_test.npy', d2)
np.save('./weights/D1_test.npy', D1)
np.save('./weights/D2_test.npy', D2)

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
X1 = whiten_matrix(center_data(d1))
X2 = whiten_matrix(center_data(d2))

# Find individual components
tolerance = 0.000001
iterations = 100
n_comp = 2

#  Implement the FastICA
S1 = fastica(X1, iterations, tolerance)
S2 = fastica(X2, iterations, tolerance)

#  Implement the FastICA
W1, H1 = nmf(D1, n_comp, iterations)
W2, H2 = nmf(D2, n_comp, iterations)

# Save the final matrices as weights
np.save('./weights/S1_test.npy', S1)
np.save('./weights/S2_test.npy', S2)
np.save('./weights/W1_test.npy', W1)
np.save('./weights/W2_test.npy', W2)

#  Write the ica files
wf.write('./out/t11_predicted_ica.wav', frq1, S1[0].astype(np.float32))
wf.write('./out/t21_predicted_ica.wav', frq1, S1[1].astype(np.float32))
wf.write('./out/t12_predicted_ica.wav', frq2, S2[0].astype(np.float32))
wf.write('./out/t22_predicted_ica.wav', frq2, S2[1].astype(np.float32))

#  Write the nmf files
wf.write('./out/t11_predicted_nmf.wav', frq1, W1[0].astype(np.float32))
wf.write('./out/t21_predicted_nmf.wav', frq1, W1[1].astype(np.float32))
wf.write('./out/t12_predicted_nmf.wav', frq2, W2[0].astype(np.float32))
wf.write('./out/t22_predicted_nmf.wav', frq2, W2[1].astype(np.float32))

# Calculate the Mean Square Error
# Formula Reference : https://www.geeksforgeeks.org/python-mean-squared-error/
MSE1 = np.square(np.subtract(sound1, S1[0]+ center_data(sound1))).mean()
MSE2 = np.square(np.subtract(sound2, S1[1]+ center_data(sound2))).mean()
MSE3 = np.square(np.subtract(sound4, S2[0]+ center_data(sound4))).mean()
MSE4 = np.square(np.subtract(sound3, S2[1]+ center_data(sound3))).mean()
MSE5 = np.square(np.subtract(sound1, W1[0]+ center_data(sound1))).mean()
MSE6 = np.square(np.subtract(sound2, W1[1]+ center_data(sound2))).mean()
MSE7 = np.square(np.subtract(sound4, W2[0]+ center_data(sound4))).mean()
MSE8 = np.square(np.subtract(sound3, W2[1]+ center_data(sound3))).mean()

# Print the Mean Square Error Values
with open('./out/MSE.txt', 'w') as f:
    sys.stdout = f 
    print("The MSE1 value is: ", MSE1)
    print("The MSE2 value is: ", MSE2)
    print("The MSE3 value is: ", MSE3)
    print("The MSE4 value is: ", MSE4)
    print("The MSE5 value is: ", MSE5)
    print("The MSE6 value is: ", MSE6)
    print("The MSE7 value is: ", MSE7)
    print("The MSE8 value is: ", MSE8)
    sys.stdout = sys.stdout


# Plot the Predicted vs Actual data
plot_wav_test([sound1, S1[0]+ center_data(sound1)], 'Male (6:45) ICA')
plot_wav_test([sound2, S1[1]+ center_data(sound2)], 'Female (6:45) ICA')
plot_wav_test([sound4, S2[0]+ center_data(sound4)], 'Male (0:20) ICA')
plot_wav_test([sound3, S2[1]+ center_data(sound3)], 'Female (0:20) ICA')
plot_wav_test([sound1, W1[0] + center_data(sound1)], 'Male (6:45) NMF')
plot_wav_test([sound2, W1[1] + center_data(sound2)], 'Female (6:45) NMF')
plot_wav_test([sound4, W2[0] + center_data(sound4)], 'Male (0:20) NMF')
plot_wav_test([sound3, W2[1] + center_data(sound3)], 'Female (0:20) NMF')



