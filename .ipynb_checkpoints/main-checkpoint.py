# Imports 
import numpy as np
import matplotlib.pyplot as plt
from util import *
from scipy.io import wavfile as wf

np.random.seed(0)

# Main source file paths
path = './audio/'
files = ['mix1.wav', 'mix2.wav', 'mix3.wav']

# initialize signals array
ica_signals = []
nmf_signals = []
frq = 0
times = 0

# Open and plot the wav signals while storing the data in a matrix
for f in files: 
    frq, data, times = read_wav(path + f)
    plot_wav(data, times, f)
    d1 = center_data(data)
    d2 = scale_data(data)
    ica_signals.append(d1)
    nmf_signals.append(d2)

# Convert array into a matrix of stacked rows
d1 = np.vstack(ica_signals)
d2 = np.vstack(nmf_signals)

# Save the matrices as checkpoint weights
np.save('./weights/d1.npy', d1)
np.save('./weights/d2.npy', d2)

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
X = whiten_matrix(d1)

# Find individual components
tolerance = 0.000001
iterations = 100
num_components = 3

# Calculate the FastICA separation
S = fastica(X, iterations, tolerance)

# Calculate the NMF separation
W, H = nmf(d2, num_components, iterations)

# Calculate the Final results as matrice weights
np.save('./weights/S.npy', S)
np.save('./weights/W.npy', W)

# Write the predicted separated sounds for ICA
wf.write('./out/s1_predicted_ica.wav', frq, S[0].astype(np.float32))
wf.write('./out/s2_predicted_ica.wav', frq, S[1].astype(np.float32))
wf.write('./out/s3_predicted_ica.wav', frq, S[2].astype(np.float32))

# Write the predicted separated sounds for NMF
wf.write('./out/s1_predicted_nmf.wav', frq, W[0].astype(np.float32))
wf.write('./out/s2_predicted_nmf.wav', frq, W[1].astype(np.float32))
wf.write('./out/s3_predicted_nmf.wav', frq, W[2].astype(np.float32))

# Plot the graphs
plot_wav(S[0], times, 'S1 ICA')
plot_wav(S[1], times, 'S2 ICA')
plot_wav(S[2], times, 'S3 ICA')
plot_wav(W[0], times, 'S1 NMF')
plot_wav(W[1], times, 'S2 NMF')
plot_wav(W[2], times, 'S3 NMF')
