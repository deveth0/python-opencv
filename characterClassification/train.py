import argparse
import glob
import os
import random
import shutil
from functools import partial
from multiprocessing.dummy import Pool
from pathlib import Path
from subprocess import call

import cv2 as cv
import numpy as np
from common import *


# region copy from https://github.com/opencv/opencv/blob/master/samples/python
# License: https://github.com/opencv/opencv/blob/master/LICENSE
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)


class KNearest(StatModel):
    def __init__(self, k=3):
        self.k = k
        self.model = cv.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, self.k)
        return results.ravel()


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv.ml.SVM_RBF)
        self.model.setType(cv.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


# endregion

def evaluate_model(_model, _digits, _samples, _labels):
    resp = _model.predict(_samples)
    err = (_labels != resp).mean()
    print("error: %.2f %%" % (err * 100))

    tiles = []
    misses = 0
    for img, flag in zip(_digits, resp == _labels):
        _img = np.copy(img)
        if not flag:
            _img[..., :2] = 0
            misses += 1
        tiles.append(_img)

    print("Model has missed {0:d} of {1:d} tests".format(misses, len(_digits)))
    return mosaic(tiles)


def create_background_crops(num_samples):
    list_backgrounds = glob.glob(GEN_DATA_DIR + "/*.jpg", recursive=False)
    print("Generating background crops from {0:d} backgrounds".format(len(list_backgrounds)))
    for i in range(0, num_samples):
        bg_img = cv.imread(random.choice(list_backgrounds))
        image_height, image_width, _ = bg_img.shape
        x = random.randint(0, image_height - CHAR_SIZE)
        y = random.randint(0, image_width - CHAR_SIZE)
        if bool(random.getrandbits(1)):
            bg_img = cv.bitwise_not(bg_img)
        cv.imwrite(DATA_DIR + "/crops/crop_" + str(i) + ".jpg", bg_img[x:x + CHAR_SIZE, y:y + CHAR_SIZE])


def create_data(num_samples):
    if os.path.exists(DATA_DIR + "/crops"):
        shutil.rmtree(DATA_DIR + "/crops")
    if os.path.exists(DATA_DIR + "/chars"):
        shutil.rmtree(DATA_DIR + "/chars")
    os.makedirs(DATA_DIR + "/crops")
    os.makedirs(DATA_DIR + "/chars")

    create_background_crops(num_samples)

    pool = Pool(os.cpu_count())
    commands = []

    fonts = glob.glob(GEN_DATA_DIR + "/*.ttf", recursive=False)
    backgrounds = glob.glob(DATA_DIR + "/crops/crop_*.jpg", recursive=False)
    textcolors = ["white", "black", "red", "green", "blue"]
    # Create images
    for idx, char in enumerate(CHARS):
        char_output_dir = DATA_DIR + "/chars/" + str(char) + "/"
        if not os.path.exists(char_output_dir):
            os.makedirs(char_output_dir)
        for j in range(0, num_samples):
            font = random.choice(fonts)
            background = random.choice(backgrounds)
            color = random.choice(textcolors)
            distort = get_distort_arg()
            blur = random.randint(0, 1)
            noise = random.randint(0, 1)
            x = str(random.randint(-3, 3))
            y = str(random.randint(-3, 3))
            commands.append("magick convert " + str(background) + " -fill " + color + " -font " + \
                            str(font) + " -weight 200 -pointsize 24 -distort Perspective " + str(distort) + " " + "-gravity center" + " -blur 0x" + str(blur) \
                            + " -evaluate Gaussian-noise " + str(noise) + " " + " -annotate +" + x + "+" + y + " " + char + " " + char_output_dir + "output_file" + str(idx) + str(j) + ".jpg")

    for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
        if returncode != 0:
            print("%d command failed: %d" % (i, returncode))


def get_distort_arg():
    amount = 5
    hundred_minus_amount = 100 - amount
    return "\"0,0 " + str(np.random.randint(0, amount)) + "," + str(np.random.randint(0, amount)) + " 100,0 " + str(
        np.random.randint(hundred_minus_amount, 100)) + "," + str(np.random.randint(0, amount)) + " 0,100 " + str(np.random.randint(0, amount)) + "," + str(
        np.random.randint(hundred_minus_amount, 100)) + " 100,100 " + str(np.random.randint(hundred_minus_amount, 100)) + "," + str(
        np.random.randint(hundred_minus_amount, 100)) + "\""


def load_data():
    """Load the data set from DATA_DIR"""
    _digits, _labels = [], []
    _imgs = glob.iglob(DATA_DIR + "/chars/**/*.jpg", recursive=True)
    for _img in _imgs:
        if os.path.isfile(_img):
            _digit = cv.imread(_img)  # , cv.IMREAD_GRAYSCALE)
            if _digit is not None:
                _digits.append(_digit)
                _path = Path(_img)
                _labels.append(ord(_path.parent.name))
    return np.array(_digits), np.array(_labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Character Detection.")
    parser.add_argument("-s", "--samples", dest="samples", type=int, help="Number of samples to generate (if not set, no images are generated and existing data from gen_data/ is used)")
    parser.add_argument("--headless", dest="headless", action='store_true', help="Run headless (don't display any images)")
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    if args.samples:
        print("Generating data")
        create_data(args.samples)

    print("Loading data")
    chars, labels = load_data()
    print("Preprocessing data")
    rand = np.random.RandomState()
    shuffle = rand.permutation(len(chars))
    chars, labels = chars[shuffle], labels[shuffle]
    samples = preprocess_hog(chars)

    num_train = int(0.9 * len(chars))
    print("Found {0:d} samples, using {1:d} for training".format(len(chars), num_train))

    chars_train, chars_test = np.split(chars, [num_train])
    labels_train, labels_test = np.split(labels, [num_train])
    samples_train, samples_test = np.split(samples, [num_train])

    print("Training KNearest")
    model = KNearest()
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, chars_test, samples_test, labels_test)
    if not args.headless:
        cv.imshow("KNearest test", vis)

    # Read more about SVMs: https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72
    print("training SVM...")
    # C: Regularization, influcences how much to avoid misclassifying by choosing smaller-margin planes if required
    # Gamma: high gamma will only consider nearby points of the resulting plane. 
    #        If a low gamma gives better results, this means that your data points are sparse
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, chars_test, samples_test, labels_test)
    if not args.headless:
        cv.imshow("SVM test", vis)

    print("saving SVM as " + MODEL_FILE + "...")
    model.save(MODEL_FILE)
    print("done")

    cv.waitKey(0)
