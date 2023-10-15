import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def gaussFilter(img_in, ksize, sigma):
    # Make gaussian kernel using function from convo.py
    from convo import make_kernel  # DIY kernel
    kernel = make_kernel(ksize,sigma)

    # Convolve the image with the gaussian kernel and return the filtered image
    convolved_img = convolve(img_in, kernel, mode = "constant")

    # from convo import slow_convolve # DIY convolution
    #convolved_img = slow_convolve(img_in, kernel) # DIY convolution

    return kernel, convolved_img.astype(int)


def sobel(img_in):
    # kernel equal to gx kernel
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Convolve the image with gx and gy (which is the transpose of gx and flipped)
    gx = convolve(img_in, kernel, mode="constant")
    gy = convolve(img_in, np.flip(kernel.T), mode="constant")

    # from convo import slow_convolve # DIY convolution
    # gx = slow_convolve(img_in, kernel) # DIY convolution
    # gy = slow_convolve(img_in, np.flip(kernel.T)) # DIY convolution
    return gx.astype(int), gy.astype(int)


def gradientAndDirection(gx, gy):
    # g equal to (gx²+ gy²)^0.5
    g = (gx**2 + gy**2)**0.5
    # Check for negative values in g, as this might indicate an integer overflow (^0.5 will probably fail beforehand, but anyway)
    if np.amin(g) < 0: print("Integer overflow")
    # theta (gradient direction) is the arctan2 of gy,gx
    theta = np.arctan2(gy, gx)
    return g.astype(int), theta


def convertAngle(angle):
    # Degrees are equal to radians times 180/pi (radians / pi = amount of half circles)
    degree = angle * 180 / np.pi
    # Convert degree to 0-180. (degree / 180) % 1 is equal to the decimal part of the division. This could be shortened with substitution.
    converted_degree = (degree / 180) % 1 * 180
    # Round degree to nearest 45 degrees (with 180 = 0). int(.. + 0.5) instead round(), as round() will round half to even and not up.
    rounded_degree = int(converted_degree / 45 + 0.5) * 45
    if rounded_degree == 180:
        rounded_degree = 0
    return rounded_degree


def maxSuppress(g, theta):
    # Empty max_sup image with same dimensions as g
    max_sup = np.zeros_like(g)
    # For each pixel, excluding border pixels, find local maxima in each gradient direction
    for y in range(1, max_sup.shape[0] - 1):
        for x in range(1, max_sup.shape[1] - 1):
            # Convert radians of theta to (0,45,90,135) degrees
            degree = convertAngle(theta[y, x])
            # Depending on the degree, find the local maxima in the corresponding gradient direction
            if degree == 0:
                if (g[y, x] >= g[y, x + 1]) and (g[y, x] >= g[y, x - 1]):
                    max_sup[y, x] = g[y, x]
            elif degree == 45:
                if (g[y, x] >= g[y + 1, x - 1]) and (g[y, x] >= g[y - 1, x + 1]):
                    max_sup[y, x] = g[y, x]
            elif degree == 90:
                if (g[y, x] >= g[y - 1, x]) and (g[y, x] >= g[y + 1, x]):
                    max_sup[y, x] = g[y, x]
            elif degree == 135:
                if (g[y, x] >= g[y + 1, x + 1]) and (g[y, x] >= g[y - 1, x - 1]):
                    max_sup[y, x] = g[y, x]
    return max_sup


def hysteris(max_sup, t_low, t_high):
    # For each pixel in max_sup, check if magnitude is below t_low, between t_low and t_high, or above t_high
    for y in range(max_sup.shape[0]):
        for x in range(max_sup.shape[1]):
            if max_sup[y, x] <= t_low:
                max_sup[y, x] = 0
            elif (max_sup[y, x] > t_low) and (max_sup[y, x] <= t_high):
                max_sup[y, x] = 1
            elif (max_sup[y, x] > t_high):
                max_sup[y, x] = 2

    # For each pixel in max_sup that is 2, set it to 255 and check surrounding pixels for 1s
    for y in range(max_sup.shape[0]):
        for x in range(max_sup.shape[1]):
            # If the magnitude is above t_high
            if max_sup[y, x] == 2:
                max_sup[y, x] = 255
                # For each surrounding pixel (3x3)
                for y2 in range(y-1, y+1):
                    for x2 in range(x-1, x+1):
                        # Check if the surrounding pixel is not beyond the image border
                        if (y2 >= 0) and (y2 <= max_sup.shape[0]-1) and (x2 >= 0) and (y2 <= max_sup.shape[1]-1):
                            # If the surrounding pixel is within the image and is equal to 1, set it to 255 as well
                            if max_sup[y2, x2] == 1:
                                max_sup[y2, x2] = 255
    return max_sup


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
