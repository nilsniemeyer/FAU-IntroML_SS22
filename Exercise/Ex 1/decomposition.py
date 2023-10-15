import numpy as np
import matplotlib.pyplot as plt


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    plt.style.use('ggplot')
    xt = []
    t = np.linspace(0, 1, samples, endpoint=True)
    for i in t:
        sum = 0
        for k in range(k_max):
            sum += ((-1)**k*(np.sin(2*np.pi*(2*k+1)*frequency*i)/((2*k+1)**2)))
        xt.append((8/(np.pi**2))*sum)
    plt.figure(figsize=(4, 1))
    plt.plot(t,xt)
    plt.show()
    return np.array(xt)


def createSquareSignal(samples: int, frequency: int, k_max: int):
    plt.style.use('ggplot')
    xt = []
    t = np.linspace(0, 1, samples, endpoint=True)
    for i in t:
        sum = 0
        for k in range(1,k_max+1):
            sum += np.sin(2*np.pi*(2*k-1)*frequency*i)/(2*k-1)
        xt.append((4/np.pi)*sum)
    plt.figure(figsize=(4, 1))
    plt.plot(t,xt)
    plt.show()
    return np.array(xt)


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    plt.style.use('ggplot')
    xt = []
    t = np.linspace(0, 1, samples, endpoint=True)
    for i in t:
        sum = 0
        for k in range(1,k_max+1):
            sum += np.sin(2*np.pi*k*frequency*i)/k
        xt.append((amplitude/2)-((amplitude/np.pi)*sum))
    plt.figure(figsize=(4, 1))
    plt.plot(t,xt)
    plt.show()
    return np.array(xt)
