# 08 - Racoon Dataset (Colab Notebook)
# Reference: https://github.com/datitran/raccoon_dataset
# Article: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9

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

# Import modules (chess detection):
import skimage
import numpy as np
from skimage import io, transform
import os
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import urllib.request
import urllib.error

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append('..')
from object_detection.utils import ops as utils_ops

# Upgrade GTF
!pip install tensorflow --upgrade
# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Set paths
PATH_TO_API = '/content/models/research/object_detection'
PATH_TO_REPO = '/content/models/research/object_detection/raccoon_dataset'
PATH_TO_DATA = '/content/models/research/object_detection/raccoon_dataset/data'
PATH_TO_LABELS = '/content/models/research/object_detection/raccoon_dataset/annotations'

!echo 'git clone TF_ObjectDetection_API'
!echo '===> cd $PATH_TO_API'
%cd $PATH_TO_API
!echo ''
!echo '===> ls $PATH_TO_API'
!ls -al $PATH_TO_API
!echo ''
!rm -rf ./TF_ObjectDetection_API
!git clone https://github.com/walteryu/raccoon_dataset
!echo ''
!echo '===> cd $PATH_TO_REPO'
%cd $PATH_TO_REPO
!echo ''
!echo '===> ls $PATH_TO_REPO'
!ls -al $PATH_TO_REPO
!echo ''

# Create TF record
!echo '===> xml_to_csv.py'
!python xml_to_csv.py
!echo ''
!echo '===> generate_tfrecord.py'
!python generate_tfrecord.py
!echo ''
