import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


MEAN = [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]

W_MEAN = [[0.5, 1, 0.5],
          [1, 2, 1],
          [0.5, 1, 0.5]]

SORBEL_X = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]

SORBEL_Y = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]

PREWITT_X = [[-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]]

PREWITT_Y = [[1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]]


def gaussian_function(x, y, o):
    power = -1 * ((x**2)+(y**2)) / (2 * (o**2))
    result = (1/(2 * math.pi * (o**2))) * (math.e**power)
    return result


def gaussian_filter(n, std):
    guass_filter = [[0 for i in range(n)] for j in range(n)]
    mid_point = n//2
    for i in range(n):
        for j in range(n):
            guass_filter[i][j] = gaussian_function(j-mid_point, i-mid_point, std)
    return guass_filter

def neighbouring_pixels(n, x, y, im):
    neighbour = []
    im_y = len(im)
    im_x = len(im[0])
    size = []

    current_y = y-1
    for i in range(n):
        current_x = x - 1
        temp = []
        for j in range(n):
            if (current_x < 0) or (current_y < 0) or \
                    (current_x >= im_x) or (current_y >= im_y):
                temp.append(0)
            else:
                temp.append(im[current_y, current_x])
                size.append([i, j])
            current_x += 1
        neighbour.append(temp)
        current_y += 1
    return neighbour, size


def convolution(im, name="temp", gray=True, filter=MEAN, smoothing=False, write=True, isImage=True):
    if isImage:
        im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    n = len(filter)
    convoluted_image = [[0 for i in range(len(im[0]))] for j in range(len(im))]
    flat_filter = [x for i in filter for x in i]
    for i in range(len(im)):
        for j in range(len(im[0])):
            division = sum(flat_filter)
            image, size = neighbouring_pixels(n, j, i, im)
            image = [x for i in image for x in i]
            if len(size) < (n**2):
                division = sum([filter[i[0]][i[1]] for i in size])
            if smoothing:
                convoluted_image[i][j] = (round(sum(
                    [a*b for a, b in zip(image, flat_filter)])/division))
            else:
                convoluted_image[i][j] = (round(sum(
                    [a * b for a, b in zip(image, flat_filter)])))
    if write:
        cv2.imwrite(name, np.array(convoluted_image))
    return np.array(convoluted_image)


def edge_strength(image_a, image_b, name="temp", write=True, isImage=True):
    if isImage:
        image_a = cv2.imread(image_a, cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(image_b, cv2.IMREAD_GRAYSCALE)
    edge_str = []
    for i in range(len(image_a)):
        edge_str.append([math.sqrt((a**2)+(b**2)) for a, b in zip(image_a[i], image_b[i])])
    if write:
        cv2.imwrite(name, np.array(edge_str))
    return np.array(edge_str)


def difference(image_a, image_b, name="temp", write=True, isImage=True):
    if isImage:
        image_a = cv2.imread(image_a, cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(image_b, cv2.IMREAD_GRAYSCALE)
    diff = []
    for i in range(len(image_a)):
        diff.append([abs(int(a)-int(b)) for a, b in zip(image_a[i], image_b[i])])
    if write:
        cv2.imwrite(name, np.array(diff))
    return np.array(diff)


def thresholding(image, threshold_a, name="temp",  threshold_b=255, write=True, isImage=True):
    if isImage:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    thresh = []
    for i in range(len(image)):
        thresh.append([255 if (a > threshold_a) and (a <= threshold_b) else 0 for a in image[i]])
    if write:
        cv2.imwrite(name, np.array(thresh))
    return np.array(thresh)


def adaptive_thresholding(im, threshold, name="temp", isImage=True, t_filter=W_MEAN):
    image = im
    if isImage:
        image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    image = convolution(image, filter=t_filter, name="guass.bmp", smoothing=True, write=True, isImage=False)
    true_image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    corrected_bg = difference(true_image, image, name="adapt_diff.bmp", isImage=False)
    results = thresholding(corrected_bg, threshold, write=False, isImage=False)
    cv2.imwrite(name, np.array(results))


def plot_threshold_histogram(im):
    image = [x for i in cv2.imread(im, cv2.IMREAD_GRAYSCALE) for x in i]
    bin = [a for a in range(0, 256)]
    plt.hist(image, bin)
    plt.title("Frequency of Pixels between Pixel Values in the range of 0 - 255")
    plt.xlabel("Pixel Values")
    plt.ylabel("Pixel Frequency")
    plt.show()
