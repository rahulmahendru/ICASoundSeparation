import wave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


''' 
Convert the given audio mixed file into float data
input: File name
Return: Frequency, time, data
'''
def read_wav(file):
    mix = wave.open(file, 'r')                                        # Open File
    sample_freq = mix.getframerate()                                  # Get Framerate
    n_samples = mix.getnframes()                                      # Get frames
    data = mix.readframes(n_samples)                                  # Convert to binary data
    data = np.frombuffer(data, dtype=np.int16)                        # Convert to float 
    times = np.linspace(0, n_samples/sample_freq, num=n_samples)      # Get the time for the file for plotting
    return sample_freq, data, times


''' 
Plot the audio data
input: data, filename
'''
def plot_wav(data, time, f):
    plt.figure(figsize=(15,5))
    plt.plot(time, data)
    plt.title('Audio mixture ' + f)
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.savefig('./img/'+f+'.png')


''' 
Plot the test audio data
input: data, filename
'''
def plot_wav_test(data, f):
    plt.figure(figsize=(15,5))
    plt.plot(data[0], alpha=0.5)
    plt.plot(data[1], alpha=0.5)
    plt.title('Audio mixture ' + f)
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.savefig('./img/'+f+'.png')


''' 
Center the data by subtracting the mean 
input: data
Return: centered data
'''
def center_data(data):
    d = data - np.mean(data)
    return d


''' 
Scale the data between 0 and 1
input: data
Return: scaled data
Formula Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
'''
def scale_data(data):
    max = 1
    min = 0
    data = np.array(data, dtype=np.float64)
    data_std = (data - np.min(data)) / (np.max(data) - np.min(data))
    data_scaled = data_std * (max - min) + min
    return data_scaled


''' 
Whiten the matrix using the formula E*D^(-1/2)*E^T*x
input: data
Return: whitened data
'''
def whiten_matrix(d):
    cov = np.dot(d, d.T)/d.shape[1]                          # Calculate the covariance matrix
    w, v = np.linalg.eigh(cov)                               # Eigenvalue decomposition of the covariance matrix
    d_w = np.diag(w)                                         # Calculate diagonal of eigenvalues
    inv_d_w = np.sqrt(np.linalg.pinv(d_w))                   # Computing D^(-1/2) using Moore-Penrose pseudo-inverse
    white_space = np.dot(v, np.dot(inv_d_w, v.T))            # Calculating PCA space E*D^{1/2}*E^T
    whiteMatrix = np.dot(white_space, d)                     # Project onto PCA -> PCA_Space * x
    # print(np.dot(whiteMatrix, whiteMatrix.T)/d.shape[1])   # Make sure the covariance is an identity matrix
    return whiteMatrix


''' 
Calculate g for negentropy
input: u
Return: tanh(u)
'''
def g(u):
    return np.tanh(u)


''' 
Calculate g' for negentropy
input: u
Return: 1 - (g(u)^2)
'''
def gd(u):
    return 1 - (g(u) ** 2)


''' 
Calculate the objective function for maximizing negentropy E{xg(w^Tx)} - E{g'(w^Tx)}
input: w, x
Return: objective function 
Reference: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e (did not copy code, used to understand the implementation of negentropy and fastICA as described in the FastICA wikipedia page)
'''
def obj(w, x):
    first_term = (x * g(np.dot(w.T, x)).T).mean(axis=1)     # Calculate E{xg(w^Tx)}
    second_term = gd(np.dot(w.T, x)).mean() * w             # Calculate E{g'(w^Tx)

    temp = first_term - second_term
    temp /= np.sqrt((temp ** 2).sum())

    return temp


''' 
Implement the fastICA algorithm 
input: X, iterations, tolerance
Return: Individual components S
'''
def fastica(X, iterations, tolerance=0.000005):
    n = X.shape[0]
    w = np.zeros((n, n), dtype=X.dtype)                             # Initialize weights 
    
    for i in np.arange(n):
        wi = np.random.rand(n)                                      # Initialize a random weight

        for j in np.arange(iterations):
            temp = obj(wi, X)                                       # Calculate the objective function
            if i > 0:
                temp -= np.dot(np.dot(temp, w[:i].T), w[:i])        # Update the demixing matrix
            dis = np.abs(np.abs((wi * temp).sum()) - 1)             # Calculate the distance for convergence
            wi = temp
            if(dis < tolerance):
                break
    
        w[i, :] = wi                                                # Update the weight
    S = np.dot(w, X)                                                # Compute the Individual signals
    
    return S


''' 
Implement the KL-NMF algorithm 
input: X, number of components, iterations
Return: W (basis vectors), H (activation) 
Algorithm blueprint reference: https://ccrma.stanford.edu/~njb/teaching/sstutorial/part2.pdf
'''
def nmf(X, k, iter):
    X = X.T
    X = normalize(X)              
    H = np.random.rand(k, X.shape[1])
    W = np.random.rand(X.shape[0], k)
    W = normalize(W)

    ones = np.ones(X.shape)

    for i in range(iter):
        H *= ((W.T @ (X / (W @ H))) / (W.T @ ones))                    # Update Activations
        W *= (((X / (W @ H)) @ H.T) / (ones @ H.T))                    # Update dictionaries
        H = np.nan_to_num(H)                                           # Convert sparse values to 0
        W = np.nan_to_num(W)
        W = normalize(W)

    return W.T, H.T
