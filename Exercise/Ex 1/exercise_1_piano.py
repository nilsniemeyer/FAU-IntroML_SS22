import matplotlib.pyplot as plt
import numpy as np


def load_sample(filename, duration=4*44100, offset=44100//10):
    signal = np.load(filename)

    starting = np.argmax(signal)+offset
    ending = starting + duration

    if ending > len(signal):
        signal = np.pad(signal, (0, ending-len(signal)), 'constant')

    signal = signal[range(starting, ending)]

    plt.plot(signal)
    plt.show()

    return signal


def compute_frequency(signal, min_freq=20):
    y = abs(np.fft.rfft(signal))/len(signal)

    frequency = np.arange(len(signal) // 2 + 1) / (len(signal)/44100)

    y = y[frequency > min_freq]
    frequency = frequency[frequency > min_freq]

    plt.plot(frequency, y)
    plt.show()

    return frequency[np.argmax(y)]


if __name__ == '__main__':
    print("A2 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A2.npy")))
    print("A3 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A3.npy")))
    print("A4 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A4.npy")))
    print("A5 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A5.npy")))
    print("A6 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A6.npy")))
    print("A7 frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.A7.npy")))
    print("Unknown frequency: ", compute_frequency(load_sample("./sounds/Piano.ff.XX.npy")))