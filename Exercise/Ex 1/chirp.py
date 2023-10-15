import numpy as np
import matplotlib.pyplot as plt


def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    plt.style.use('ggplot')
    xt = []
    t = np.linspace(0, duration, duration * samplingrate, endpoint=True)
    if linear:
        plt.suptitle("Linear chirp")
        c = (freqto - freqfrom) / duration
        for i in t:
            xt.append(np.sin(2 * np.pi * (freqfrom + c / 2 * i) * i))
    else:
        plt.suptitle("Exponential chirp")
        k = (freqto / freqfrom) ** (1 / duration)
        for i in t:
            xt.append(np.sin(((2 * np.pi * freqfrom) / (np.log(k))) * (k ** i - 1)))
    plt.plot(t, xt)
    plt.show()
    return np.array(xt)
