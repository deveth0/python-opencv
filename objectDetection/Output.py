import cv2
import numpy as np


class Output:

    def __init__(self, classes):
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))
        self.classes = classes

    def draw_prediction(self, image, class_id, confidence, x1, y1, x2, y2):
        image_height, image_width, _ = image.shape
        fontSize = round(image_height / 1024)
        label = (str(self.classes[class_id]) + " " + str(round(confidence * 100)) + "%").upper()
        color = self.colors[class_id]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, fontSize)
        (textWidth, textHeight), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontSize)
        if y1 - textHeight <= 0:
            y1 = y1 + textHeight
            cv2.rectangle(image, (x1, y1), (x1 + textWidth, y1 - textHeight), color, -1)
            cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontSize)
        else:
            cv2.rectangle(image, (x1, y1), (x1 + textWidth, y1 - textHeight), color, -1)
            cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontSize)
