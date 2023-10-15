from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize,ksize))

    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            # Calculate x and y values for the gaussian formula as distances from the centre
            gaussian_x = abs((ksize//2)-x)
            gaussian_y = abs((ksize//2)-y)
            kernel[y, x] = (1/(2*np.pi*sigma**2)) * np.exp(-(gaussian_x**2+gaussian_y**2)/(2*sigma**2))

    kernel = (1 / np.sum(kernel)) * kernel
    return kernel


def slow_convolve(arr, k):
    # New empty array
    convolved_arr = np.zeros_like(arr)

    # Pad existing array
    # Pad left and right
    pad_leftright = k.shape[1] // 2
    padded_arr = np.pad(arr, [(0, 0), (pad_leftright, pad_leftright)], mode='constant')
    # Pad up and down
    pad_updown = k.shape[0] // 2
    padded_arr = np.pad(padded_arr, [(pad_updown, pad_updown), (0, 0)], mode='constant')

    # Mirror kernel
    k = np.flip(k)

    # Convolution
    for y in range(convolved_arr.shape[0]):
        for x in range(convolved_arr.shape[1]):
            neighbours = padded_arr[y:y + k.shape[0], x:x + k.shape[1]]
            convolved_arr[y, x] = np.sum(neighbours * k)

    return convolved_arr


if __name__ == '__main__':
    k = make_kernel(9, 5)

    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    plt.imshow(im)
    plt.show()

    convolved_img = np.zeros_like(im)

    for channel in (0, 1, 2):
        convolved_img[:, :, channel] = slow_convolve(im[:, :, channel], k)

    plt.imshow(convolved_img)
    plt.show()

    result = im.astype('int16') + (im.astype('int16') - convolved_img.astype('int16'))

    for channel in (0, 1, 2):
        result[result[:, :, channel] > 255] = 255
        result[result[:, :, channel] < 0] = 0

    plt.imshow(result)
    plt.show()

    



