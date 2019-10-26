import argparse

import cv2

from YoloObjectDetector import YoloObjectDetector

parser = argparse.ArgumentParser(description="Detect Objects in Image.")
parser.add_argument("-s", "--size", dest="size", default="320", type=str, help="Spatial Size (tiny, 320, 416, 608), default: 320")
parser.add_argument("-i", "--input", dest="input", type=str, help="Input file (optional). If not set, Webcam will be used.")
args = parser.parse_args()

modelSize = args.size
print("Using size " + modelSize)

detector = YoloObjectDetector(modelSize)
cam = None

if args.input is None:
    cam = cv2.VideoCapture(0)
    _, image = cam.read()
else:
    image = cv2.imread(args.input)

detector.processImage(image)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1024, 1024)
cv2.imshow('image', image)

if cam is not None:
    cam.release()
cv2.waitKey(5000)
cv2.destroyAllWindows()
