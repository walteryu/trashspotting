# 14 - Hedgehog Object Detection Tutorial
# Article: https://medium.com/@dana.yu/training-a-custom-object-detection-model-41093ddc5797
# Github: https://github.com/danamyu/hedgehog_detector

# Reference: https://github.com/wagonhelm/TF_ObjectDetection_API
# Clone tensorflow/model repo into colab root dir, /content:
!git clone --quiet https://github.com/tensorflow/models.git
!apt-get install -qq protobuf-compiler python-tk
!pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
!pip install -q pycocotools

# Config protoc, slim and builder script:
%cd /content/models/research
!protoc object_detection/protos/*.proto --python_out=.
import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
!python object_detection/builders/model_builder_test.py

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

# This is needed to display the images.
%matplotlib inline

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Set paths
PATH_TO_API = '/content/models/research/object_detection'
PATH_TO_UTILS = '/content/models/research/object_detection/utils'
# PATH_TO_REPO = '/content/models/research/object_detection/TF_ObjectDetection_API'
# PATH_TO_DATA = '/content/models/research/object_detection/TF_ObjectDetection_API/data'
# PATH_TO_LABELS = '/content/models/research/object_detection/TF_ObjectDetection_API/labels'
PATH_TO_REPO = '/content/models/research/object_detection/hedgehog_detector'
PATH_TO_MODEL = '/content/models/research/object_detection/hedgehog_detector/models'
PATH_TO_RE = '/content/models/research/object_detection/hedgehog_detector/models/research'
PATH_TO_OBJECT = '/content/models/research/object_detection/hedgehog_detector/models/research/object_detection'
PATH_TO_TRAIN = '/content/models/research/object_detection/hedgehog_detector/train_object_detector'

# !echo 'git clone TF_ObjectDetection_API'
# !echo '===> cd $PATH_TO_API'
# %cd $PATH_TO_API
# !rm -rf ./TF_ObjectDetection_API
# !git clone https://github.com/walteryu/TF_ObjectDetection_API.git

!echo 'git clone hedgehog_detector'
!echo '===> cd $PATH_TO_API'
%cd $PATH_TO_API
!rm -rf ./hedgehog_detector
!git clone https://github.com/danamyu/hedgehog_detector.git

# !echo ''
# !echo '===> cd $PATH_TO_REPO'
# %cd $PATH_TO_REPO
# !echo ''
# !echo '===> ls $PATH_TO_REPO'
# !ls -al $PATH_TO_REPO
# !echo ''

!echo ''
!echo '===> cd $PATH_TO_RE'
%cd $PATH_TO_RE
!echo ''
!echo '===> ls $PATH_TO_RE'
!ls -al $PATH_TO_RE
!echo ''

!echo ''
!echo '===> cp config'
!cp ./object_detection/samples/configs/ssd_mobilenet_v1_coco.config ./
!ls -al $PATH_TO_OBJECT
!echo ''

# Train model
%cd $PATH_TO_RE
!python object_detection/train.py \
        --logtostderr \
        --train_dir=train_object_detector/Labels/hedgehog \
        --pipeline_config_path=SSD_mobilenet_v1_coco.config

# --pipeline_config_path=models/research/object_detection/samples/configs/SSD_mobilenet_v1_coco.config
