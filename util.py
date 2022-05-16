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
    return data, times

def plot_wav(data, times, f):
    plt.figure(figsize=(15,5))
    plt.plot(times, data)
    plt.title('Audio mixture '+f)
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.show()
