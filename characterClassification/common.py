import itertools
from math import sqrt

import cv2 as cv
import numpy as np
from numpy.linalg import norm

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
        "0123456789"
CHAR_SIZE = 32  # size of a single character (32px x 32px)
DATA_DIR = "target"  # folder where generated images are placed
GEN_DATA_DIR = "gen_data"  # folder where sources for generation can be found (fonts etc)
MODEL_FILE = DATA_DIR + "/chars_svm.dat"


# region copy from https://github.com/opencv/opencv/blob/master/samples/python
# License: https://github.com/opencv/opencv/blob/master/LICENSE
def mosaic(imgs, w=0):
    if w == 0:
        w = int(sqrt(len(imgs)))
    imgs = iter(imgs)
    img0 = next(imgs)
    pad = np.zeros_like(img0)
    imgs = itertools.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * CHAR_SIZE * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (CHAR_SIZE, CHAR_SIZE), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    output = itertools.zip_longest(fillvalue=fillvalue, *args)
    return output


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

# endregion
