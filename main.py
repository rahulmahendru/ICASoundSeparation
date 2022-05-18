# Imports 
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from util import *

# Source file paths
path = './audio/'
files = ['mix1.wav', 'mix2.wav', 'mix3.wav']

# initialize signals array
signals = []

# Open and plot the wav signals while storing the data in a matrix
for f in files: 
    data, times = read_wav(path + f)
    # plot_wav(data, times, f)
    d = center_data(data)
    signals.append(d)

# Convert array into a matrix of stacked rows
d = np.vstack(signals)

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
X = whiten_matrix(d)

# Find individual components
Y = []
tolerance = 0.000001

# Implement the FastICA algorithm
fastica(X, Y, tolerance)
    