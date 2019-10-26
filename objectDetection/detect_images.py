import argparse
import glob
import os, shutil


import cv2

from YoloObjectDetector import YoloObjectDetector
parser = argparse.ArgumentParser(description="Detect Objects in all images in the given folder.")
parser.add_argument("-s", "--size", dest="size", default="320", type=str, help="Spatial Size (tiny, 320, 416, 608), default: 320")
parser.add_argument("input", type=str, help="Input folder.")
args = parser.parse_args()

modelSize = args.size
print("Using size " + modelSize)

detector = YoloObjectDetector(modelSize)

srcFolder = args.input
print("Using imagefolder " + srcFolder)

if os.path.exists("target"):
    shutil.rmtree("target")

os.mkdir("target")
files = glob.glob(srcFolder + "/**/*.jpg", recursive=True)
numberFiles = len(files)
for idx, filename in enumerate(files):
    print("{0:3.2f}% ({1:d}/{2:d})".format((100/numberFiles)*idx, idx, numberFiles))
    if os.path.isfile(filename):  # filter dirs
        print(filename)
        image = cv2.imread(filename)

        image = detector.processImage(image)
        if image is not None:
            output = os.path.join("target", os.path.basename(filename))
            cv2.imwrite(output, image)
