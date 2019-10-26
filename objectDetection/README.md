# Object Detection using a pretrained Model

This sample project uses the [YOLOv3](https://pjreddie.com/darknet/yolo/) model to detect different types of objects (80 classes, defined by the [COCO Dataset](http://cocodataset.org/)).

![Output](https://github.com/deveth0/python-opencv/blob/master/objectDetection/site/birds_output.png?raw=true "Output")

First of all you need to download the two different weight files and place them in the `models/yolo*` folders. This can be done automatically using `downloadYOLO.sh`.

## Usage
Each script allows you to pass a spatial size as parameter. The following values can be used:

* tiny
* 320 (default)
* 416
* 608

Depending on the choosen size, the detection might run quite slow. Tiny uses a different model which is optimized for small devices. 

Example:
````bash
python3 SCRIPT.py -s tiny
````

You can exit each script by pressing `q` and if you require additional help run it with the `-h` flag.

## detect_image.py and detect_video.py

Those scripts use either a Webcam or a File as input and show the result of the detection. If you do not pass a file as input, the script tries to use your webcam. You can find some example images in the `data/` directory.

Example:
````bash
# Use webcam
python3 detect_image.py
# Use image
python3 detect_image.py -i data/cat1.jpg
````

## detect_images.py

This script iterates through all images in the given folder, performs an object detection on them and writes the result into `target/`. 

Example:
````bash
# All images in date/
python3 detect_images.py data/
````
