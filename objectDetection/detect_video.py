import argparse
import cv2

from YoloObjectDetector import YoloObjectDetector

parser = argparse.ArgumentParser(description="Detect Objects in a Video.")
parser.add_argument("-s", "--size", dest="size", default="320", type=str, help="Spatial Size (tiny, 320, 416, 608), default: 320")
parser.add_argument("-i", "--input", dest="input", type=str, help="Input file (optional). If not set, Webcam will be used.")
args = parser.parse_args()

modelSize = args.size
print("Using size " + modelSize)

detector = YoloObjectDetector(modelSize)

if args.input is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.input)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    detector.processImage(frame)

    # show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(5000)
cv2.destroyAllWindows()
