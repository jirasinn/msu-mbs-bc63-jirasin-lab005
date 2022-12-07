ARG cuda_version=10.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

LABEL maintainer="zinmoehtoo2015@gmail.com"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR home/projects/facial_recognition
RUN apt-get update
RUN apt-get install -y \
    libopencv-dev \
    python3-pip \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    python3-setuptools
RUN pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD . .
CMD python3 -c 'import torch; print(torch.__version__)'