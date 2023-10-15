import numpy as np
import matplotlib.pyplot as plt


def polarToKart(shape, r, theta):
    x = r * np.cos(theta) + shape[1]/2
    y = r * np.sin(theta) + shape[0]/2
    return int(y), int(x)


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    magnitude_spectrum = np.abs(np.fft.fft2(img))
    magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)
    magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum)
    return magnitude_spectrum_db


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    print(min(magnitude_spectrum.shape)/2)
    R = np.zeros(k)
    for i in range(1,k+1):
        for theta in np.linspace(0, np.pi, sampling_steps, endpoint=True):
            for r in range(k*(i-1),k*i+1):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                R[i-1] += magnitude_spectrum[y, x]
    return R

def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    max_ray = min(magnitude_spectrum.shape)//2
    T = np.zeros(k)
    for i in range(1, k + 1):
        for theta in np.linspace((i-1), i, sampling_steps-1, endpoint=True):
            theta = theta * np.pi / k
            for r in range(0, max_ray):
                y, x = polarToKart(magnitude_spectrum.shape, r, theta)
                T[i - 1] += magnitude_spectrum[y, x]
    return T



def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    magnitude_spectrum = calculateMagnitudeSpectrum(img)
    #plt.subplot(2, 2, 1)
    #plt.imshow(magnitude_spectrum, 'gray')
    #plt.title('Magnitude Spectrum')

    # Extract ring features
    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    # Extract fan features
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)
    plt.show()
    return R, T


