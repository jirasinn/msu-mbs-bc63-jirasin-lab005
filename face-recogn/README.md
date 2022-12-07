# Face Recognition System 

## Introduction

Realtime face recognition pipeline using pytorch. Here, [mtcnn](https://arxiv.org/abs/1604.02878) is used for face detection and [facenet](https://ieeexplore.ieee.org/document/7298682) to extract feature embeddings.

## Requirements

- Pytorch >= 1.0
- [facenet_pytorch](https://github.com/timesler/facenet-pytorch)
- scikit-learn
- pillow
- numpy 
- opencv-python

## Quick Start 

### `Step 1` (Collect faces)

**`Support Mode`**
- webcam
- image or 
- video

```bash
#with webcam
python3 collect_face.py --mode webcam --interval 15 --total-image 30 --save-path datasets/users

```
With this command, you need to type user name. After it, just press *`'esc'`* to close the webcam.
**Beware** Don't click the [x] button on webcam frame. This will crash the system from opencv. 
**NOTE** : This user name will create a folder for training stage.

```bash
#with images
python3 collect_face.py --mode image --image-path 'datasets/' --save-path datasets/users

```
**NOTE** : Before using above command, you need to collect sample images in a folder with the respective user name.

### `Step 2` (Train recognizer)

U can choose the classifier to be used grid_search.
After training finished, `the user features`, `classifier` and `name_pair` are stored in a *recognizer.pkl* file.

```bash
python3 train.py --dataset 'datasets/users' --save_model 'recognizer.pkl' --grid_search

```

### `Step 3` (Testing the model)

**`Support Mode`**
- webcam
- image or 
- video

```bash
python3 infer_demo.py --mode 'webcam' --confidence 0.6 --model 'recognizer.pkl'

```


## Using Docker

**NOTE** : To use nvidia-docker, you first need to install `nvidia-driver` in your host machine.
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo apt-get install nvidia-driver-440
reboot
```
After restarted, check in terminal using this command ``nvidia-smi`` will show the gpu driver info.
Then:

- Install docker-ce from [here](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/#os-requirements)
- Install nvidia-docker v1

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

- Test nvidia-docker installation

```bash
docker run --gpus all nvidia/cuda:10.0-cudnn7-devel nvidia-smi

```

- Build the *Dockerfile* for face_recognition testing. This will take time.

```bash
docker build -t face_recognition:v1 .

```
- Test the Docker Image

```
xhost +local:root
docker run --rm -it --gpus all --env="DISPLAY" --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix --ipc=host face_recognition:v1.1 bash

```

Then start use the command with **Quick Start** steps.














