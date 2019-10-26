#!/bin/sh

set -e

wget https://pjreddie.com/media/files/yolov3.weights -O models/yolo/yolov3.weights

wget https://pjreddie.com/media/files/yolov3-tiny.weights -O models/yolo-tiny/yolov3-tiny.weights
