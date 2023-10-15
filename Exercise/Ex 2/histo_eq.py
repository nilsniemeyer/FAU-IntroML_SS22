from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def empty_pdf():
    # Generate array with intensities 0-255 and zero counts
    intensity = np.arange(256)
    counts_zero = np.zeros(256)
    pdf_empty = np.stack((intensity, counts_zero), axis=-1)
    return pdf_empty


def get_pdf(image):
    # Generate pdf from counts of intensities in image
    # Set pdf as array returned from empty_pdf
    pdf = empty_pdf()
    # Put PIL image object into array
    im_array = np.array(image)
    # Change zeros to counts of corresponding intensity
    for i in range(256):
        pdf[i, 1] = (im_array == i).sum()
    # Plot histogram of PDF with counts
    plt.bar(pdf[:, 0], pdf[:, 1])
    plt.show()
    # Change counts to percentages for later use
    pdf[:, 1] /= im_array.size
    return pdf


def get_cdf(image):
    # Generate CDF from PDF
    # Get PDF for image
    pdf = get_pdf(image)
    # Set cdf as array returned from empty_pdf (same x-axis)
    cdf = empty_pdf()
    # Calculate cumulative sum. Every value in CDF is equal to previous sum plus new pdf value
    for i in range(256):
        cdf[i, 1] = cdf[i - 1, 1] + pdf[i, 1]
    # Plot and return CDF
    plt.plot(cdf[:, 0], cdf[:, 1])
    plt.show()
    return cdf


def equalization(image):
    # Perform histogram equalization
    # Get CDF for image
    cdf = get_cdf(image)
    # Set Cmin as minimum value of all nonzero values in CDF
    cdf_nonzero = cdf[np.flatnonzero(cdf[:, 1]), 1]
    Cmin = cdf_nonzero[0]
    # Loop over every pixel in image
    for x in range(image.width):
        for y in range(image.height):
            # Get old pixel intensity, calculate and set new pixel intensity
            pixel_intensity_old = image.getpixel((x, y))
            pixel_intensity_new = (round((cdf[pixel_intensity_old,1] - Cmin) / (1 - Cmin) * 255))
            image.putpixel((x, y), pixel_intensity_new)
    return image


# Load and show image
im = Image.open("hello.png")
im.show()
# Perform equalization and show histogram equalized image
im_equalized = equalization(im)
im_equalized.show()
# Show PDF and CDF of equalized histogram
get_cdf(im_equalized)
