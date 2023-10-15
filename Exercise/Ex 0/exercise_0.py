import numpy as np
import matplotlib.pyplot as plt


def clip(arr, min, max):
    for i in range(len(arr)):
        if arr[i] < min:
            arr[i] = min
        elif arr[i] > max:
            arr[i] = max
    return arr


if __name__ == '__main__':
    array = np.random.rand(100)
    result = clip(array, 0.2, 0.8)
    plt.plot(array, result, '.')
    plt.show()
