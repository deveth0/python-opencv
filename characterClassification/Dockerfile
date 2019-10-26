FROM python:3.7-buster

RUN apt-get update \
    && apt-get -y install wget build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://imagemagick.org/download/ImageMagick.tar.gz \
    && tar xvzf ImageMagick.tar.gz \
    && rm ImageMagick.tar.gz \
    && cd ImageMagick-7* \
    && ./configure \
    && make -j4 \
    && make install \
    && ldconfig /usr/local/lib \
    && pip3 install opencv-python numpy
COPY . /
ENTRYPOINT ["python3", "train.py", "--headless"]