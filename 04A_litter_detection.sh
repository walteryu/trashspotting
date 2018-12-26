# 04A - KAB Litter Detection Algorithm
# Reference: https://github.com/isaychris/litter-detection-tensorflow

# # Clone tensorflow/model repo into /content dir:
# !echo '===> clone models into /content (same as starting dir):'
# !git clone --quiet https://github.com/tensorflow/models.git
# !echo ''
# !echo '===> install packages into system dir'
# !apt-get install -qq protobuf-compiler python-tk
# !pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
# !pip install -q pycocotools
# !echo ''
#
# # Manage all directories closely by absolute path from root:
# !echo '===> Current dir:'
# !pwd
# !echo ''
# !echo '===> Current dir contents:'
# !ls -al ./
# !echo ''
# !echo '===> ls /content/models'
# %cd /content/models
# !echo ''
# !echo '===> ls /content/models'
# !ls -al /content/models
# !echo ''
# !echo '===> cd /content/models/research'
# %cd /content/models/research
# !echo ''
# !echo '===> ls /content/models/research'
# !ls -al /content/models/research
# !echo ''
#
# # Config protoc, slim and builder script:
# !protoc object_detection/protos/*.proto --python_out=.
# import os
# os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
# !python object_detection/builders/model_builder_test.py
#
# import numpy as np
# import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import tensorflow as tf
# import zipfile
# import math
# import time
#
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image
#
# # This is needed since the notebook is stored in the object_detection folder.
# # sys.path.append("..")
# sys.path.append('..')
# from object_detection.utils import ops as utils_ops
#
# # Upgrade GTF
# !pip install tensorflow --upgrade
#
# # if tf.__version__ < '1.4.0':
# #   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
#
# # This is needed to display the images.
# %matplotlib inline
#
# # Here are the imports from the object detection module.
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# Clone repo into object_detection dir
PATH_TO_OBJECT_DET = '/content/models/research/object_detection'
!echo '===> cd $PATH_TO_OBJECT_DET'
%cd $PATH_TO_OBJECT_DET
!echo ''
!echo '===> ls $PATH_TO_OBJECT_DET'
!ls -al $PATH_TO_OBJECT_DET
!echo ''

# # Reference: https://github.com/isaychris/litter-detection-tensorflow
# !echo 'git clone litter-detection-tensorflow'
# # !rm -rf ./litter-detection-tensorflow
# !git clone https://github.com/isaychris/litter-detection-tensorflow.git
# !echo ''

!echo 'git clone trashspotting'
!rm -rf ./trashspotting
!git clone https://github.com/walteryu/trashspotting.git
!echo ''
PATH_TO_TRASHSPOTTING = '/content/models/research/object_detection'
!echo '===> cd $PATH_TO_TRASHSPOTTING'
%cd $PATH_TO_TRASHSPOTTING
!echo ''
!echo '===> ls $PATH_TO_TRASHSPOTTING'
!ls -al $PATH_TO_TRASHSPOTTING
!echo ''

# >>>>> START HERE FOR AFTER LOADING MODULES >>>>>
# Change dir to run model
PATH_TO_REPO = '/content/models/research/object_detection/litter-detection-tensorflow'
!echo '===> cd $PATH_TO_REPO'
%cd $PATH_TO_REPO
!echo ''
!echo '===> ls $PATH_TO_REPO'
!ls -al $PATH_TO_REPO
!echo ''

# What model to use
MODEL_NAME = 'litter_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph/frozen_inference_graph.pb'
PATH_TO_MODEL = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph/saved_model'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('kab_training', 'litter_detection_map.pbtxt')
# PATH_TO_LABELS = os.path.join('config', 'litter_detection_map.pbtxt')
PATH_TO_LABELS = '/content/models/research/object_detection/litter-detection-tensorflow/config/litter_detection_map.pbtxt'

NUM_CLASSES = 1

config = tf.ConfigProto()
config.gpu_options.force_gpu_compatible = True
config.gpu_options.per_process_gpu_memory_fraction = 1

# Reference: https://gist.github.com/Arafatk/c063bddb9b8d17a037695d748db4f592
# from google.protobuf import text_format
# from tensorflow.python.platform import gfile
#
# def pbtxt_to_graphdef(filename):
#   with open(filename, 'r') as f:
#     graph_def = tf.GraphDef()
#     file_content = f.read()
#     text_format.Merge(file_content, graph_def)
#     tf.import_graph_def(graph_def, name='')
#     tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pb', as_text=False)
#
# def graphdef_to_pbtxt(filename):
#   with gfile.FastGFile(filename,'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     tf.import_graph_def(graph_def, name='')
#     tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)
#   return
#
# !echo '===> cd /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph'
# %cd /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph
# !echo ''
# !echo '===> ls /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph'
# !ls -al /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph
# !echo ''
# pbtxt_to_graphdef('/content/models/research/object_detection/litter-detection-tensorflow/config/litter_detection_map.pbtxt')
# !echo '===> ls /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph'
# !ls -al /content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph
# !echo ''

# # Load a (frozen) Tensorflow model into memory.
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#     serialized_graph = fid.read()
#
#     # Reference: https://stackoverflow.com/questions/35351760/tf-save-restore-graph-fails-at-tf-graphdef-parsefromstring
#     # from google.protobuf import text_format
#     # graph_def = tf.GraphDef()
#     # text_format.Merge(proto_b, graph_def)
#     # text_format.Merge(serialized_graph, graph_def)
#
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')

# def openGraph():
#   graph = tf.Graph()
#   graphDef = tf.GraphDef()
#   with open(PATH_TO_CKPT, "rb") as graphFile:
#     graphDef.ParseFromString(graphFile.read())
#   with graph.as_default():
#     tf.import_graph_def(graphDef)
#   return graph
#   # graph = openGraph()
# openGraph()

!echo '===> cd $PATH_TO_TRASHSPOTTING'
%cd $PATH_TO_TRASHSPOTTING
!echo ''
!python freeze_graph.py --$PATH_TO_MODEL
