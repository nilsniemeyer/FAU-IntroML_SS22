import numpy as np

def create_greyscale_histogram(img):
    hist = np.zeros(256)
    # Change zeros to counts of corresponding intensity
    for i in range(256):
        hist[i] = (img == i).sum()
    return hist


def binarize_threshold(img, t):
    # Set as 0 if below or equal to threshold, set as 255 if above
    img[img <= t] = 0
    img[img > t] = 255
    return img


def p_helper(hist, theta: int):
    # p0 equal to sum of histogram values from 0 to threshold
    # p1 equal to sum of histogram values from threshold+1 to end of histogram
    p0 = np.sum(hist[range(0, theta + 1)])
    p1 = np.sum(hist[range(theta + 1, len(hist))])
    return p0, p1


def mu_helper(hist, theta, p0, p1):
    # mu0 equal to 1 divided by p0 times the sum of (intensities times probabilities) from 0 to threshold
    # mu0 equal to 1 divided by p1 times the sum of (intensities times probabilities) from threshold+1 to end
    mu0 = 1 / p0 * sum(np.multiply(range(0, theta + 1), hist[range(0, theta + 1)]))
    mu1 = 1 / p1 * sum(np.multiply(range(theta + 1, len(hist)), hist[range(theta + 1, len(hist))]))
    return mu0, mu1


def calculate_otsu_threshold(hist):
    # Initiate maximal sigma squared and corresponding optimal theta
    sig_squared_max = 0.0
    theta_opt = 0.0
    # From absolute values to percentages/probabilities
    hist /= np.sum(hist)
    # Calculate sigma squared for each possible threshold
    # Possible thresholds are between (including) first and (excluding) last nonzero value in histogram
    for theta in range(np.amin(np.flatnonzero(hist)),np.amax(np.flatnonzero(hist))):
        # Calculate p0,p1,mu0,mu1 for given histogram and threshold
        p0, p1 = p_helper(hist, theta)
        mu0, mu1 = mu_helper(hist, theta, p0, p1)
        # Calculate sigma squared
        sig_squared = p0 * p1 * ((mu1 - mu0) ** 2)
        # Update maximal sigma squared and optimal theta
        if sig_squared > sig_squared_max:
            sig_squared_max = sig_squared
            theta_opt = theta
    return theta_opt


def otsu(img):
    # Load image into array
    im_array = np.array(img)
    # Get histogram
    hist = create_greyscale_histogram(im_array)
    # Calculate optimal threshold for histogram
    theta = calculate_otsu_threshold(hist)
    # Binarize the image with optimal threshold
    im_array_binary = binarize_threshold(im_array, theta)
    return im_array_binary
