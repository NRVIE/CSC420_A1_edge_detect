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

def compute_t_i(img, t):
    sum_of_low = 0
    low_count = 0
    sum_of_upper = 0
    upper_count = 0
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            curr_pix = img[row][col]
            if (curr_pix <= t):
                sum_of_low += curr_pix
                low_count += 1
            else:
                sum_of_upper += curr_pix
                upper_count += 1
    ml = sum_of_low / low_count
    mh = sum_of_upper / upper_count
    return (ml + mh)/2

def threshold(gray_img, epsilon = 0.01):
    """Assume input gray_img is a gray scale image"""
    img = gradient(gray_img)
    # init threshold
    new_img = np.zeros((img.shape[0], img.shape[1]))
    t0 = np.sum(img) / (img.shape[0] * img.shape[1])
    t_i = compute_t_i(img, t0)
    while ((t_i - t0) > epsilon):
        t0 = t_i
        t_i = compute_t_i(img, t0)
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            if img[row][col] >= t_i:
                new_img[row][col] = 255
            else:
                new_img[row][col] = 0
    return new_img


img = cv2.imread('image1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Q7
img_gray_noise1 = add_gnoise(img_gray, 0.0, 0.1)
img_gray_noise2 = add_gnoise(img_gray, 0.0, 0.3)
# print(type(img_gray_noise1))
# print(type(img_gray_noise2))
# edges1 = cv2.Canny(img_gray_noise1, threshold1=75, threshold2=100)
# edges2 = cv2.Canny(img_gray_noise2, threshold1=75, threshold2=100)

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(img_gray_noise1, cmap='gray')
ax1.title.set_text('Noise with variance=0.1'), ax1.set_xticks([]), ax1.set_yticks([])
ax2.imshow(img_gray_noise2, cmap='gray')
ax2.title.set_text('Noise with variance=0.3'), ax2.set_xticks([]), ax2.set_yticks([])

# # Q8
# print(type(img))
# threshold_img = threshold(img_gray)
#
# plt.figure(figsize=(7, 7))
# plt.imshow(threshold_img, cmap='gray')
# plt.xticks([]), plt.yticks([])
