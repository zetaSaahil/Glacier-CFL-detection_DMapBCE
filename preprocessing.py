from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from scipy.ndimage.measurements import variance
import scipy as sp
import imageio
import matplotlib.pyplot as plt
import cv2

def preprocess():
    name = ''
    return name

class Filter():
    def __init__(self):
        pass

    def lee_filter(self, img, size):
        img_mean = uniform_filter(img, size)
        img_sqr_mean = uniform_filter(img**2, size)
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = variance(img)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (img - img_mean)
        return img_output

    def kuan_filter(self, img, size=11, no_looks=8.0):
        M,N = img.shape
        i = 0
        new_image = np.zeros_like(img)
        while i<=M-size:
            j = 0
            while j<=N-size:
                window_img = img[i:i+size, j:j+size]
                w_mean = np.mean(window_img)
                w_std = np.std(window_img)
                ci = w_std/(w_mean + np.finfo(np.float32).eps)
                Cu = np.sqrt(1 / no_looks)
                W = (1 - Cu*Cu / (ci*ci+ np.finfo(np.float32).eps)) / (1 + Cu*Cu)
                new_image[int(np.floor(i+size/2)), int(np.floor(j+size/2))] = img[int(np.floor(i+size/2)), int(np.floor(j+size/2))]*W + w_mean*(1-W)
                j+=1
            i+=1
        return new_image

    def enhanced_lee(self, img, size=7, no_looks=4, damping_factor=1.5):
        M, N = img.shape
        i = 0
        new_image = np.zeros_like(img)
        while i <= M - size:
            j = 0
            while j <= N - size:
                window_img = img[i:i + size, j:j + size]
                l_m = np.mean(window_img)
                sd = np.std(window_img)
                Cu = 1/np.sqrt(no_looks)
                Cmax = np.sqrt(1 + 2/no_looks)
                ci = sd/(l_m + np.finfo(np.float32).eps)
                K = np.exp(- damping_factor*(ci - Cu) + np.finfo(np.float32).eps / (Cmax - ci+ np.finfo(np.float32).eps))
                if ci <= Cu:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = l_m
                elif Cu <= ci <= Cmax:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = l_m * K + img[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] * (1 - K)
                elif ci >= Cmax:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = img[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))]
                j += 1
            i += 1
        return new_image

    def frost_filter(self, img, size, damping_factor=1.0):
        M, N = img.shape
        print(M,N)
        i = 0
        new_image = np.zeros_like(img)
        while i <= M - size:
            j = 0
            while j <= N - size:
                window_img = img[i:i + size, j:j + size]
                S = np.zeros_like(window_img)
                centrepix= int(np.floor(size/2))
                for k in range(size):
                    for l in range(size):
                        S[k,l] = np.sqrt(np.square(k - centrepix) + np.square(l - centrepix))
                l_m = np.mean(window_img)
                l_v = np.var(window_img)
                B = damping_factor * (l_v / l_m * l_m)
                K = np.exp(-B*S + np.finfo(np.float32).eps)
                new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = np.sum(window_img*K)/np.sum(K)
                j += 1
                print(i,j)
            i += 1
        return new_image

    def gamma(self, img, size=7, no_oflooks=3):
        M, N = img.shape
        print(M, N)
        i = 0
        new_image = np.zeros_like(img)
        while i <= M - size:
            j = 0
            while j <= N - size:
                window_img = img[i:i + size, j:j + size]
                centrepix = int(np.floor(size / 2))
                Im = np.mean(window_img)
                S = np.std(window_img)
                Cu = np.sqrt(1/no_oflooks)
                Cmax = np.sqrt(2)*Cu
                Ci = S/(Im+ np.finfo(np.float32).eps)
                A = (1+Cu*Cu)/(Ci*Ci - Cu*Cu + np.finfo(np.float32).eps)
                B = A-no_oflooks-1
                D = Im*Im*B*B + 4*A*no_oflooks*Im*centrepix
                Rf = (B*Im + np.sqrt(D))/(2*A)
                if Ci <= Cu:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = Im
                elif Cu < Ci < Cmax:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = Rf
                elif Ci >= Cmax:
                    new_image[int(np.floor(i + size / 2)), int(np.floor(j + size / 2))] = centrepix
                j += 1
            i += 1
        return new_image



