import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy import signal


def add_gnoise(img, mean: float, sigma: float):
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))
    return img + gaussian


def gaussian(x: int, sigma: float, mean: float = 0.0):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        (-1 / 2) * ((x - mean) ** 2 / (sigma ** 2)))


def create_gfilter(size: int, sigma: float):
    # calculating the x-value of each index of 1D-Gaussian matrix
    # then we can use the x-value of each index to get the value of filter for gaussian function
    g_vector = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        g_vector[i] = gaussian(g_vector[i], sigma)
    # Use outer product to get 2D-Gaussian filter
    return np.outer(g_vector, g_vector)


def convolve_3b3(img, filter):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    for i_row in range(1, img.shape[0]):
        for i_col in range(1, img.shape[1]):
            if i_row == 0 or i_col == 0 or ((i_row + 1) % img.shape[0]) == 0 or (
                    (i_col + 1) % img.shape[1]) == 0:
                new_img[i_row][i_col] = img[i_row][i_col]
                continue
            sum_of_pix = 0
            for f_row in [2, 1, 0]:
                for f_col in [2, 1, 0]:
                    sum_of_pix += filter[f_row][f_col] * img[i_row - f_row + 1][i_col - f_col + 1]
            new_img[i_row][i_col] = sum_of_pix
    return new_img


def gradient(img):
    sobel_x = np.outer([1, 2, 1], [-1, 0, 1])
    sobel_y = np.outer([-1, 0, 1], [1, 2, 1])
    gx = convolve_3b3(img, sobel_x)
    gy = convolve_3b3(img, sobel_y)
    return np.sqrt(gx ** 2 + gy ** 2)


img = cv2.imread('image1.jpg')
print(type(img))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gradient_img = gradient(img_gray)

plt.figure(figsize=(7, 7))
plt.imshow(gradient_img, cmap='gray')
plt.xticks([]), plt.yticks([])
