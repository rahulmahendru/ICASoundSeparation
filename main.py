# Imports 
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from util import *

# Source file paths
path = './audio/'
files = ['mix1.wav', 'mix2.wav', 'mix3.wav']

# Open and plot the wav signals while storing the data in a matrix
for f in files: 
    data, times = read_wav(path+f)
    plot_wav(data, times, f)
