# Character Classification with a synthetic data set

This sample project creates a dataset of characters (A-Z, 0-9) with some random parameters (like font color or blur) and uses it to train two different models (KNearest and Support Vector Machine/SVM). 

To validate the accuracy of the models, 10% of the generated data is used for testing and 90% for training. After running the training, you'll see the two test datasets (one for KNearest, one for SVM) and all characters that where not classified correctly are marked in red.

![Output](https://github.com/deveth0/python-opencv/blob/master/characterClassification/site/SVM_output.png?raw=true "Output")

For further usage the SVM model is saved in `target/chars_svm.dat` and can be used in your own code:

````python
import cv2 as cv
import os
import sys

saved_model = "target/chars_svm.dat"
if not os.path.exists(saved_model):
    sys.exit('"%s" not found, run train.py first' % saved_model)

model = cv.ml.SVM_load(saved_model)
````

## Data Generation

First of all you need to generate a data set and use it for training. This is done by the `train.py` script. If you pass a `-s` parameter, this defines the number of samples that is created for each character in training (e.g. if you use `-s 100` you'll get 100 images of A, 100 images of B etc).

As the generation takes quite a long time, you can skip it for later runs if you want by obmitting the `-s` parameter.

Example:
````bash
# Generate 36*100 images and train the model
python3 train.py -s 100 
# Reuse dataset and train the model
python3 train.py 
````

The image creation requires `ImageMagick 7` to be installed. If you don't want or can't install it on your system, you can also use a Docker image for training (see [Docker](#Docker)).

All sources for the image creation can be found in `gen_data` where you can also add new backgrounds and fonts (or replace the existing ones):

* `*.jpg`: `jpg` files are used to create crops (32px x 32px) on which the characters are placed
* `*.ttf`: `ttf` files define the font(s) that are used to create characters



## Docker

If you have a running Docker installation, you can also use the Docker Image [deveth0/opencv-characterdetection](https://cloud.docker.com/u/deveth0/repository/docker/deveth0/opencv-characterclassification). It includes all the background images and fonts from this repository. 

To run the image, you need to create a `target/` folder and then run the image with a `bind` mount. When the training is complete, you can find both the created dataset and the saved SVM model in `target/`.

````bash
# Run docker image with training size of 1000
mkdir target
docker run --mount type=bind,source="$(pwd)"/target,target=/target deveth0/opencv-characterclassification "-s 1000"
````

To replace the `gen_data` folder in the image, you can mount a local one into the container:
````bash
# Run docker image with training size of 1000
mkdir target
docker run --mount type=bind,source="$(pwd)"/target,target=/target --mount type=bind,source="$(pwd)"/gen_data,target=/gen_data deveth0/opencv-characterclassification "-s 1000"
````


## Acknowledgments

The idea to create a syntetic data set originates from [Github: learnopencv - CharClassification](https://github.com/spmallick/learnopencv/tree/master/CharClassification) and is used in a modified way in this project.

## Literature

If you want to find out more about how a Support Vector Machine works, I highly recommend this [article](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72) by [Savan Patel](https://github.com/savanpatel).
