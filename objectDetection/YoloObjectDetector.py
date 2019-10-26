import cv2
import numpy as np
import os
import sys

from Output import Output


class YoloObjectDetector:

    def __init__(self, modelSize):
        if not (os.path.exists("models/yolo-tiny/yolov3-tiny.weights") and os.path.exists("models/yolo/yolov3.weights")):
            sys.exit("Please download the YOLOv3 weight files first (using downloadYOLO.sh)")
        if modelSize == "tiny":
            self.modelSize = 416
            self.model = cv2.dnn.readNet("models/yolo-tiny/yolov3-tiny.weights",
                                         "models/yolo-tiny/yolov3-tiny.cfg")

        else:
            self.modelSize = int(modelSize)
            self.model = cv2.dnn.readNet("models/yolo/yolov3.weights",
                                         "models/yolo/yolov3.cfg")

        with open("models/yolo/yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.output = Output(self.classes)
        # scale is 0.00392 for YOLO as it does not use 0..255 but 0..1 as range (0.00392 = 1/255)
        self.scale = 0.00392

    def processImage(self, image):
        if image is None:
            print("Ignoring image")
            return

        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, self.scale, (self.modelSize, self.modelSize), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        retval = self.model.forward(self.get_output_layers(self.model))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in retval:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(indices) == 0:
            return

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.output.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        return image

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
