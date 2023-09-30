import numpy as np
import math
import cv2

def add_gnoise(img, mean: float, sigma: float):
    gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1]))
    return img + gaussian

def gaussian(x: int, sigma: float, mean: float = 0.0):
    return (1/(sigma*math.sqrt(2*math.pi))) * math.exp((-1/2) * ((x - mean) ** 2/(sigma ** 2)))

def create_gfilter(size: int, sigma: float):
    # calculating the x-value of each index of 1D-Gaussian matrix
    # then we can use the x-value of each index to get the value of filter for gaussian function
    g_vector = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        g_vector[i] = gaussian(g_vector[i], sigma)
    # Use outer product to get 2D-Gaussian filter
    return np.outer(g_vector, g_vector)

def gradient():
    ...
