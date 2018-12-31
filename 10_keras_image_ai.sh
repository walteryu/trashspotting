# 10 - Keras with Image AI (Colab Notebook)
# Reference: https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Prediction
# Article: https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

# Clone tensorflow/model repo into colab root dir, /content:
!git clone --quiet https://github.com/tensorflow/models.git
!apt-get install -qq protobuf-compiler python-tk
!pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
!pip install -q pycocotools

PATH_TO_HOME = '/content'
PATH_TO_RE = '/content/models/research'
PATH_TO_API = '/content/models/research/object_detection'
PATH_TO_REPO = '/content/trashspotting'
PATH_TO_AI = '/content/trashspotting/image_ai'

# Config protoc, slim and builder script:
%cd $PATH_TO_RE
!protoc object_detection/protos/*.proto --python_out=.
import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
!python object_detection/builders/model_builder_test.py

# Keras and Image AI modules:
!pip install scipy opencv-python h5py keras
!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

# Import modules (TF_ObjectDetection_API):
import numpy as np
import scipy
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

# Import modules (chess detection):
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

# FirstDetection.py
from imageai.Detection import ObjectDetection

# Clone image repo
!echo '===> cd $PATH_TO_HOME'
% cd $PATH_TO_HOME
!echo ''

# Clone repo
!rm -rf ./trashspotting
!git clone https://github.com/walteryu/trashspotting.git

!echo '===> cd $PATH_TO_AI'
% cd $PATH_TO_AI
!echo ''

# Download model
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "trash1.jpg"), output_image_path=os.path.join(execution_path , "trash1_ai.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
