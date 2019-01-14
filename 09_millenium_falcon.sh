# 09 - Millenium Falcon Example (Colab Notebook)
# Reference: https://github.com/bourdakos1/Custom-Object-Detection
# Tutorial: https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e

# Clone tensorflow/model repo into colab root dir, /content:
# !git clone --quiet https://github.com/tensorflow/models.git
!rm -rf ./models
!mkdir models
%cd /content/models
# !git clone https://github.com/bourdakos1/Custom-Object-Detection.git
!rm -rf ./Custom-Object-Detection
!git clone https://github.com/walteryu/Custom-Object-Detection.git
%cd /content/models/Custom-Object-Detection

!apt-get install -qq protobuf-compiler python-tk
!pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
!pip install -q pycocotools

# Config protoc, slim and builder script:
!protoc object_detection/protos/*.proto --python_out=.
import os
os.environ['PYTHONPATH'] += ':/content/models/Custom-Object-Detection/:/content/models/Custom-Object-Detection/slim/'
!python object_detection/builders/model_builder_test.py

# This is needed to display the images.
# %matplotlib inline

# Import modules (TF_ObjectDetection_API):
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import math
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import skimage
import numpy as np
from skimage import io, transform
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append('..')
from object_detection.utils import ops as utils_ops

# Upgrade GTF
!pip install tensorflow --upgrade

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Set paths
PATH_TO_REPO = '/content/models/Custom-Object-Detection'
PATH_TO_TRAIN = '/content/models/Custom-Object-Detection/train'
PATH_TO_API = '/content/models/Custom-Object-Detection/object_detection'
PATH_TO_UTILS = '/content/models/Custom-Object-Detection/object_detection/utils'

# Create TF records
%cd $PATH_TO_REPO
!python object_detection/create_tf_record.py

# Download and untar model; do not download locally due to repo file size limits
!wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
!tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
!mv faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* .

# Train model
!python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=faster_rcnn_resnet101.config

# Download training files; separate module after running model
!echo '===> cd $PATH_TO_TRAIN'
%cd $PATH_TO_TRAIN
!echo ''
!echo '===> ls $PATH_TO_TRAIN'
!ls -al $PATH_TO_TRAIN
!echo ''

# Relocate checkpoint files; do not run locally due to file size
# >>>>> CHANGE STEPS HERE >>>>>
!mv ./checkpoint ../
!mv ./graph.pbtxt ../
!mv ./model.ckpt-100.data-00000-of-00001 ../
!mv ./model.ckpt-100.index ../
!mv ./model.ckpt-100.meta ../

!echo '===> cd $PATH_TO_REPO'
%cd $PATH_TO_REPO
!echo ''
!echo '===> ls $PATH_TO_REPO'
!ls -al $PATH_TO_REPO
!echo ''

# Export inference graph
# >>>>> CHANGE STEPS HERE >>>>>
!python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix model.ckpt-100 \
        --output_directory output_inference_graph

# Test model!
!python object_detection/object_detection_runner.py

# Download test results
from google.colab import files
%cd $PATH_TO_REPO/output/test_images
files.download('image-1.jpg')
files.download('image-2.jpg')
files.download('image-3.jpg')
