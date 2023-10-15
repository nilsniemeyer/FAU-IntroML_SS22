import numpy as np
import cv2
import matplotlib.pyplot as plt


def drawCircle(img, x, y):
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    t = 115
    img[img <= t] = 0
    img[img > t] = 255
    blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_CONSTANT)
    return blur


def drawLargestContour(img) -> np.ndarray:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    contour_img = np.zeros(img.shape, np.uint8)
    cv2.drawContours(contour_img, [biggest_contour], 0, 255, 2)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    pre_y_values = []
    for y in range(contour_img.shape[0]):
        if contour_img[y,x] == 255:
            pre_y_values.append(y)

    y_values = []
    for item in range(1,len(pre_y_values)-2):
        if pre_y_values[item] != pre_y_values[item-1]+1:
            y_values.append(pre_y_values[item])

    return np.array(y_values[:6])


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    kx = x2
    ky = y2
    angle = np.arctan2(y2-y1, x2-x1)

    while img[int(ky), int(kx)] == 0:
        kx = kx + np.cos(angle)
        ky = ky + np.sin(angle)
    return int(ky), int(kx)


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    kx = k1[1]
    ky = k1[0]
    new_y_angle = np.arctan2(k3[0] - k1[0], k3[1] - k1[1])
    distance = ((k1[1] - k3[1]) ** 2 + (k1[0] - k3[0]) ** 2) ** 0.5

    counter = 0
    best_kx = 0
    best_ky = 0
    min_distance_to_k2 = 999

    while counter < distance:
        kx = kx + np.cos(new_y_angle)
        ky = ky + np.sin(new_y_angle)
        distance_to_k2 = ((kx - k2[1]) ** 2 + (ky - k2[0]) ** 2) ** 0.5
        if distance_to_k2 < min_distance_to_k2:
            min_distance_to_k2 = distance_to_k2
            best_kx = kx
            best_ky = ky
        counter += 1

    rotation_matrix = cv2.getRotationMatrix2D((best_ky,best_kx), angle = (new_y_angle - (np.pi/2)) * (180/np.pi), scale = 1)
    return rotation_matrix


def palmPrintAlignment(img):
    original = img.copy()

    plt.subplot(2, 3, 1)
    plt.imshow(img, 'gray')
    plt.title('Original')

    img2 = binarizeAndSmooth(img)
    plt.subplot(2, 3, 2)
    plt.imshow(img2, 'gray')
    plt.title('Binarized')

    img3 = drawLargestContour(img2)
    plt.subplot(2, 3, 3)
    plt.imshow(img3, 'gray')
    plt.title('Contour')

    x1 = int(img.shape[1] * 0.04)
    x2 = int(img.shape[0] * 0.08)
    points1 = getFingerContourIntersections(img3, x1)
    #print("First column: ", points1)
    points2 = getFingerContourIntersections(img3, x2)
    #print("Second column: ", points2)

    middlepoints = np.array([
        [x1, int((points1[0] + points1[1])/2)],
        [x2, int((points2[0] + points2[1])/2)],
        [x1, int((points1[2] + points1[3])/2)],
        [x2, int((points2[2] + points2[3])/2)],
        [x1, int((points1[4] + points1[5])/2)],
        [x2, int((points2[4] + points2[5])/2)]])

    k1 = findKPoints(img3, middlepoints[0,1], middlepoints[0,0], middlepoints[1,1], middlepoints[1,0])  # (img, y1, x1, y2, x2)
    k2 = findKPoints(img3, middlepoints[2,1], middlepoints[2,0], middlepoints[3,1], middlepoints[3,0])  # (img, y1, x1, y2, x2)
    k3 = findKPoints(img3, middlepoints[4,1], middlepoints[4,0], middlepoints[5,1], middlepoints[5,0])  # (img, y1, x1, y2, x2)

    cv2.line(img3, (middlepoints[0,0], middlepoints[0,1]), (middlepoints[1,0], middlepoints[1,1]), (120, 120, 0), 2)
    cv2.line(img3, (middlepoints[2,0], middlepoints[2,1]), (middlepoints[3,0], middlepoints[3,1]), (120, 120, 0), 2)
    cv2.line(img3, (middlepoints[4,0], middlepoints[4,1]), (middlepoints[5,0], middlepoints[5,1]), (120, 120, 0), 2)

    cv2.circle(img3, k1[::-1], 3, (120, 120, 0), 5)
    cv2.circle(img3, k2[::-1], 3, (120, 120, 0), 5)
    cv2.circle(img3, k3[::-1], 3, (120, 120, 0), 5)

    plt.subplot(2, 3, 4)
    plt.imshow(img3, 'gray')
    plt.title('Points')

    rotation_matrix = getCoordinateTransform(k1, k2, k3)
    cv2.line(img3, k1[::-1], k3[::-1], (200, 200, 0), 2)
    plt.subplot(2, 3, 5)
    plt.imshow(img3, 'gray')
    plt.title('Lines')

    rotated = cv2.warpAffine(original, rotation_matrix, (img.shape[1], img.shape[0]))
    plt.subplot(2, 3, 6)
    plt.imshow(rotated, 'gray')
    plt.title('Rotated')
    plt.show()
    return rotated

