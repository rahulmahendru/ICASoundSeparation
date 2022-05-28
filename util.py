import wave
import numpy as np
import matplotlib.pyplot as plt

def read_wav(file):
    mix = wave.open(file, 'r')
    sample_freq = mix.getframerate()
    n_samples = mix.getnframes()
    data = mix.readframes(n_samples)
    data = np.frombuffer(data, dtype=np.int16)
    times = np.linspace(0, n_samples/sample_freq, num=n_samples)
    return sample_freq, data, times


def plot_wav(data, times, f):
    plt.figure(figsize=(15,5))
    plt.plot(times, data)
    plt.title('Audio mixture ' + f)
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.show()

def center_data(data):
    d = data - np.mean(data)
    return d

# Whiten the matrix using the formula E*D^(-1/2)*E^T*x
def whiten_matrix(d):
    # Calculate the covariance matrix
    cov = np.dot(d, d.T)/d.shape[1]

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    w, v = np.linalg.eigh(cov)

    # Calculate diagonal of eigenvalues
    d_w = np.diag(w)

    # Computing D^(-1/2) using Moore-Penrose pseudo-inverse
    inv_d_w = np.sqrt(np.linalg.pinv(d_w))

    # Calculating PCA space
    white_space = np.dot(v, np.dot(inv_d_w, v.T))

    # Project onto PCA
    whiteMatrix = np.dot(white_space, d)

    # Make sure the covariance is an identity matrix
    # print(np.dot(whiteMatrix, whiteMatrix.T)/d.shape[1])

    return whiteMatrix

def g(u):
    return np.tanh(u)

def gd(u):
    return 1 - (g(u) ** 2)

def obj(w, x):
    first_term = (x * g(np.dot(w.T, x)).T).mean(axis=1)
    second_term = gd(np.dot(w.T, x)).mean() * w

    temp = first_term - second_term
    temp /= np.sqrt((temp ** 2).sum())

    return temp

def fastica(X, iterations, tolerance=0.000005):

    n = X.shape[0]
    w = np.zeros((n, n), dtype=X.dtype)
    
    for i in np.arange(n):
        wi = np.random.rand(n)

        for j in np.arange(iterations):
            temp = obj(wi, X)
            if i > 0:
                temp -= np.dot(np.dot(temp, w[:i].T), w[:i])
            dis = np.abs(np.abs((wi * temp).sum()) - 1)
            wi = temp
            if(dis < tolerance):
                break
    
        w[i, :] = wi
    S = np.dot(w, X)
    
    return S

