# # 07 - Chess Detection Algorithm (Rocker Notebook)
# # Reference: https://github.com/wagonhelm/TF_ObjectDetection_API
#
# # Install apt package manager:
# !apt update -y && apt upgrade -y
# !apt install -y wget
# !apt-get install -y git-core
#
# # Clone tensorflow/model repo into colab root dir, /content:
# # !apt-get install -y protobuf-compiler python-tk
# !git clone --quiet https://github.com/tensorflow/models.git
# !apt-get install -y protobuf-compiler python-pil python-lxml
# !pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
# !pip install -q pycocotools
#
# HOME = '/notebooks'
# MODELS = '/notebooks/models'
# RESEARCH = '/notebooks/models/research'
#
# !echo '==> ls -al $HOME'
# !ls -al $HOME
# !echo ''
# !echo '==> ls -al $MODELS'
# !ls -al $MODELS
# !echo ''
# !echo '==> ls -al $RESEARCH'
# !ls -al $RESEARCH
# !echo ''

# >>>>> START HERE >>>>>
# Config protoc, slim and builder script:
!echo '===> ls -al $RESEARCH/object_detection'
!ls -al $RESEARCH/object_detection
!echo ''
!echo '===> ls -al $RESEARCH/object_detection/protos'
!ls -al $RESEARCH/object_detection/protos
!echo ''
!echo '==> protoc /*.proto'
# !protoc /notebooks/models/research/object_detection/protos/*.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/anchor_generator.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/argmax_matcher.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/bipartite_matcher.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/box_coder.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/box_predictor.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/eval.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/faster_rcnn.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/faster_rcnn_box_coder.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/graph_rewriter.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/grid_anchor_generator.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/hyperparams.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/image_resizer.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/input_reader.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/keypoint_box_coder.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/losses.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/matcher.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/mean_stddev_box_coder.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/model.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/multiscale_anchor_generator.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/optimizer.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/pipeline.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/post_processing.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/preprocessor.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/region_similarity_calculator.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/square_box_coder.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/ssd.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/ssd_anchor_generator.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/string_int_label_map.proto --python_out=.
!protoc /notebooks/models/research/object_detection/protos/train.proto --python_out=.
# %cd $RESEARCH
# protoc object_detection/protos/*.proto --python_out=.
!echo ''

!echo '==> export PYTHONPATH'
# !echo $PYTHONPATH
# PYTHONPATH = '/usr/bin/python'
# !export $PYTHONPATH=PYTHONPATH
#
# import os
# # os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
# !set $PYTHONPATH += ':/notebooks/models/research/:/notebooks/models/research/slim/'
!export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
!echo ''

!echo '==> python model_builder_test'
# !which protoc
# %cd /usr/bin/protoc
# !ls -al ./
!python /notebooks/models/research/object_detection/builders/model_builder_test.py
!echo ''

# # Import modules (TF_ObjectDetection_API):
# import numpy as np
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import tensorflow as tf
# import zipfile
# import math
# import time
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image

# !pip install --upgrade setuptools
# !pip install -U scikit-image
# !pip install -U skimage
#
# # Import modules (chess detection):
# import skimage
# from skimage import io, transform
# import shutil
# import glob
# import pandas as pd
# import xml.etree.ElementTree as ET
# from collections import defaultdict
#
# # import urllib.request
# # import urllib.error
#
# # This is needed since the notebook is stored in the object_detection folder.
# sys.path.append('..')
# from object_detection.utils import ops as utils_ops
#
# # Upgrade GTF
# !pip install -q tensorflow --upgrade
# # if tf.__version__ < '1.4.0':
# #   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
#
# # This is needed to display the images.
# %matplotlib inline
#
# # Here are the imports from the object detection module.
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
