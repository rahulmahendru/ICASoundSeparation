# Imports 
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from util import *
from scipy.io import wavfile as wf

# Source file paths
path = './audio/'
files = ['mix1.wav', 'mix2.wav', 'mix3.wav']

# initialize signals array
signals = []
frq = 0
# Open and plot the wav signals while storing the data in a matrix
for f in files: 
    frq, data, times = read_wav(path + f)
    # plot_wav(data, times, f)
    d = center_data(data)
    signals.append(d)

# Convert array into a matrix of stacked rows
d = np.vstack(signals)

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
X = whiten_matrix(d)

# Find individual components
tolerance = 0.000001
iterations = 100

S = fastica(X, iterations, tolerance)

wf.write('s1_predicted.wav', frq, S[0].astype(np.float32))
wf.write('s2_predicted.wav', frq, S[1].astype(np.float32))
wf.write('s3_predicted.wav', frq, S[2].astype(np.float32))