# ICASoundSeparation

This project is developed as part of the Term Project for COMP8535. This documents contains the description for the various files in the repository and the instructions to run the algorithm with provided code files.

## Audio folder
This folder contains the input audio files used in the project. The mix1, mix2 and mix3 files are the mixed signals for the instruments. The male and female files are the speech source signals for the testing phase.

## Img folder
This folder contains the output graphs gathered after running the ICA and NMF algorithms on the given datasets. 

## Weights folder
This folder contains the checkpoint matrices in .npy format which store the input and output matrices used to reach the given results. 

## Out folder
This folder contains the output audio files contructed after running the algorithms. It also contains a text file containing the MSE scores gathered during the project.

## main.py
This is the python script for running the ICA and NMF algorithms on the instrumental audio data. It uses the audio files from the audio folder and outputs the graphs for the input and output sound signals and also the .wav files in the img and out folders respectively.

The following libraries must be installed for this script to work properly:
Numpy
Matplotlib.pyplot
Util.py
Scipy.io.wavfile

The script may be run using the python command:
`python main.py`

This file uses random.seed(0) in its execution for constant results. 

## test.py
This is the python script for running the ICA and NMF algorithms on the speech audio data during the testing phase. It uses the male + female audio files from the audio folder and outputs the graphs for the input and output sound signals and also the .wav files in the img and out folders respectively.

The following libraries must be installed for this script to work properly:
Numpy
Util.py
Scipy.io.wavfile
sys

The script may be run using the python command:
`python test.py`

This file uses random.seed(0) in its execution for constant results. 

## util.py
This python script contains the functions necessary for running the project. The functions include reading and plotting the wav files, preprocessing the data for ICA and NMF, the ICA and NMF algorithms and calculating the objective function of ICA. 

The following libraries must be installed for this script to work properly:
Wave
Numpy
Matplotlib.pyplot
Sklearn.preprocessing.normalize