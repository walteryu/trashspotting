# 04A - KAB Litter Detection Algorithm
# Reference: https://github.com/isaychris/litter-detection-tensorflow

# Clone tensorflow/model repo into /content dir:
!echo '===> clone models into /content (same as starting dir):'
!git clone --quiet https://github.com/tensorflow/models.git
!echo ''
!echo '===> install packages into system dir'
!apt-get install -qq protobuf-compiler python-tk
!pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
!pip install -q pycocotools
!echo ''

# Manage all directories closely by absolute path from root:
!echo '===> Current dir:'
!pwd
!echo ''
!echo '===> Current dir contents:'
!ls -al ./
!echo ''
!echo '===> ls /content/models'
%cd /content/models
!echo ''
!echo '===> ls /content/models'
!ls -al /content/models
!echo ''
!echo '===> cd /content/models/research'
%cd /content/models/research
!echo ''
!echo '===> ls /content/models/research'
!ls -al /content/models/research
!echo ''

# Config protoc, slim and builder script:
!protoc object_detection/protos/*.proto --python_out=.
import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
!python object_detection/builders/model_builder_test.py

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

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
sys.path.append('..')
from object_detection.utils import ops as utils_ops

# Upgrade GTF
!pip install tensorflow --upgrade
# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images.
%matplotlib inline

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Clone repo into object_detection dir
PATH_TO_OBJECT_DET = '/content/models/research/object_detection'
!echo '===> cd $PATH_TO_OBJECT_DET'
%cd $PATH_TO_OBJECT_DET
!echo ''
!echo '===> ls $PATH_TO_OBJECT_DET'
!ls -al $PATH_TO_OBJECT_DET
!echo ''

# Reference: https://github.com/isaychris/litter-detection-tensorflow
!echo 'git clone litter-detection-tensorflow'
# !rm -rf ./litter-detection-tensorflow
!git clone https://github.com/isaychris/litter-detection-tensorflow.git
!echo ''

!echo 'git clone trashspotting'
!rm -rf ./trashspotting
!git clone https://github.com/walteryu/trashspotting.git
!echo ''
PATH_TO_TRASHSPOTTING = '/content/models/research/object_detection/trashspotting'
!echo '===> cd $PATH_TO_TRASHSPOTTING'
%cd $PATH_TO_TRASHSPOTTING
!echo ''
!echo '===> ls $PATH_TO_TRASHSPOTTING'
!ls -al $PATH_TO_TRASHSPOTTING
!echo ''

# >>>>> START HERE FOR AFTER LOADING MODULES >>>>>
# What model to use
MODEL_NAME = 'litter_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph/frozen_inference_graph.pb'
PATH_TO_MODEL = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph/saved_model/saved_model.pb'
PATH_TO_GRAPH = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph'
PATH_TO_PIPELINE = '/content/models/research/object_detection/litter-detection-tensorflow/config/faster_rcnn_nas.config'
PATH_TO_TRN_CKPT = '/content/models/research/object_detection/litter-detection-tensorflow/litter_inference_graph/model.ckpt.data-00000-of-00001'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('kab_training', 'litter_detection_map.pbtxt')
# PATH_TO_LABELS = os.path.join('config', 'litter_detection_map.pbtxt')
PATH_TO_LABELS = '/content/models/research/object_detection/litter-detection-tensorflow/config/litter_detection_map.pbtxt'

NUM_CLASSES = 1
config = tf.ConfigProto()
config.gpu_options.force_gpu_compatible = True
config.gpu_options.per_process_gpu_memory_fraction = 1

# Attempt No.1: Use different implementation of ParseFromString function
# Reference: https://gist.github.com/Arafatk/c063bddb9b8d17a037695d748db4f592
# Error: ParseFromString error when parsing frozen model file
#
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
# %cd $PATH_TO_GRAPH
# !pbtxt_to_graphdef(PATH_TO_LABELS)

# Attempt No.2: Freeze model from graph file and pass into ParseFromString function
# Reference: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# Error: ParseFromString error when parsing frozen model file
# %cd $PATH_TO_TRASHSPOTTING
# !python freeze_graph.py --model_dir=$PATH_TO_GRAPH \
#   --output_node_name="num_detections, detection_boxes, detection_scores, detection_classes, detection_masks"
# !python freeze_graph.py --model_dir=$PATH_TO_GRAPH \
#   --output_node_name="Softmax"

# Attempt No.3: Freeze model from graph file and pass into ParseFromString function
# # Reference: https://devtalk.nvidia.com/default/topic/1028464/jetson-tx2/converting-tf-model-to-tensorrt-uff-format/
# !pip install uff
# import uff
# output_names = ['predictions/Softmax']
# # write frozen graph to file
# # with open(frozen_graph_filename, 'wb') as f:
# with open(PATH_TO_GRAPH, 'wb') as f:
#     f.write(graph_def.SerializeToString())
# f.close()
# # convert frozen graph to uff
# uff_model = uff.from_tensorflow_frozen_model(frozen_graph_filename, output_names)

# Attempt No.4: Open as graph from checkpoint files
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

# Attempt No.5: Use different implementation of ParseFromString function
# Reference: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
# def load_graph(frozen_graph_filename):
#   # We load the protobuf file from the disk and parse it to retrieve the
#   # unserialized graph_def
#   with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#       graph_def = tf.GraphDef()
#       graph_def.ParseFromString(f.read())
#   # Then, we import the graph_def into a new Graph and returns it
#   with tf.Graph().as_default() as graph:
#       # The name var will prefix every op/nodes in your graph
#       # Since we load everything in a new graph, this is not needed
#       tf.import_graph_def(graph_def, name="prefix")
#   return graph
# load_graph(PATH_TO_CKPT)

# Attempt No.6: Freeze model using Tensorflow python tools
# Reference: https://github.com/tensorflow/tensorflow/issues/5780
# bazel build tensorflow/python/tools:freeze_graph && \
#   bazel-bin/tensorflow/python/tools/freeze_graph \
#   --input_graph=tensorflow/python/tools/freezeinput/myoutput/train/graph.pbtxt \
#   --input_checkpoint=tensorflow/python/tools/freezeinput/myoutput/train/model.ckpt-12731 \
#   --output_graph=tensorflow/python/tools/freezeoutput/frozen_graph.pb --output_node_names=Softmax \
#   --input_binary=false

# Attempt No.7: Freeze model using Tensorflow python tools
# Reference: https://github.com/nheidloff/object-detection-anki-overdrive-cars#5-testing-of-the-model
# Change dir to run tools
# !echo '===> cd $PATH_TO_OBJECT_DET'
# %cd $PATH_TO_OBJECT_DET
# !echo ''
# !echo '===> ls $PATH_TO_OBJECT_DET'
# %ls -al $PATH_TO_OBJECT_DET
# !echo ''
# !python export_inference_graph.py \
#   --input_type=image_tensor \
#   --pipeline_config_path=$PATH_TO_PIPELINE \
#   --trained_checkpoint_prefix=$PATH_TO_TRN_CKPT \
#   --output_directory=$PATH_TO_GRAPH
# !echo '===> ls $PATH_TO_OBJECT_DET'
# %ls -al $PATH_TO_GRAPH
# !echo ''

# Attempt No.8: Freeze model using Tensorflow python tools
# Reference: https://medium.com/coinmonks/tensorflow-object-detection-with-custom-objects-34a2710c6de5
# !echo '===> cd $PATH_TO_OBJECT_DET'
# %cd $PATH_TO_OBJECT_DET
# !echo ''
# !echo '===> ls $PATH_TO_OBJECT_DET'
# %ls -al $PATH_TO_OBJECT_DET
# !echo ''
# python train.py --logtostderr --train_dir=training \
#   --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

# # >>>>> END HERE FOR AFTER LOADING MODULES >>>>>
#
# # Change dir to run model
# PATH_TO_REPO = '/content/models/research/object_detection/litter-detection-tensorflow'
# !echo '===> cd $PATH_TO_REPO'
# %cd $PATH_TO_REPO
# !echo ''
# !echo '===> ls $PATH_TO_REPO'
# !ls -al $PATH_TO_REPO
# !echo ''
#
# # Load a (frozen) Tensorflow model into memory.
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   # with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#   with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
#     serialized_graph = fid.read()
#
#     # Reference: https://stackoverflow.com/questions/35351760/tf-save-restore-graph-fails-at-tf-graphdef-parsefromstring
#     from google.protobuf import text_format
#     # graph_def = tf.GraphDef()
#     # text_format.Merge(proto_b, graph_def)
#     # >>>>> PARSING ERROR
#     text_format.Merge(serialized_graph, od_graph_def)
#
#     # od_graph_def.ParseFromString(serialized_graph)
#     # tf.import_graph_def(od_graph_def, name='')
#
# # Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
#
# # Helper code
# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)
#
# def rescale(im_width, im_height, image):
#   aspectRatio = im_width / im_height
#   new_width = ( 640 * aspectRatio )
#   new_height = ( new_width / aspectRatio )
#   image = image.resize((int(new_width), int(new_height)), resample=0)
#   return image
#
# # Image detection
# # The images to be tested are located in this directory
# # PATH_TO_TEST_IMAGES_DIR = 'final_presentation'
# PATH_TO_TEST_IMAGES_DIR = '/content/models/research/object_detection/litter-detection-tensorflow/test_images'
#
# TEST_IMAGE_PATHS = []
# for root, dirs, filenames in os.walk(PATH_TO_TEST_IMAGES_DIR):
#     for f in filenames:
#         file_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, f)
#         TEST_IMAGE_PATHS.append(file_path)
#
# # Size, in inches, of the output images.
# IMAGE_SIZE = (16, 12)
#
# # THRESHOLD
# THRESHOLD = 0.65 # The minimum score threshold for showing detections. default = 0.5
# MAX_BOXES = 30  # The maximum number of boxes to draw for detections. default = 30
#
# # Run inference:
# def run_inference_for_single_image(image, graph, sess):
#   # Get handles to input and output tensors
#   ops = tf.get_default_graph().get_operations()
#   all_tensor_names = {output.name for op in ops for output in op.outputs}
#   tensor_dict = {}
#
#   for key in [ 'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
#     tensor_name = key + ':0'
#     if tensor_name in all_tensor_names:
#       tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
#
#   if 'detection_masks' in tensor_dict:
#     # The following processing is only for single image
#     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks( detection_masks, detection_boxes, image.shape[0], image.shape[1])
#     detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#     # Follow the convention by adding back the batch dimension
#     tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
#
#   image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#
#   # Run inference
#   output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
#
#   # all outputs are float32 numpy arrays, so convert types as appropriate
#   output_dict['num_detections'] = int(output_dict['num_detections'][0])
#   output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
#   output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#   output_dict['detection_scores'] = output_dict['detection_scores'][0]
#
#   if 'detection_masks' in output_dict:
#     output_dict['detection_masks'] = output_dict['detection_masks'][0]
#
#   return output_dict
#
# # Run algorithm
# print('\n// Running object detection algorithm')
#
# with detection_graph.as_default():
#   sess = tf.Session(graph=detection_graph,config=config)
#   with tf.device('/device:GPU:0'):
#     for i, image_path in enumerate(TEST_IMAGE_PATHS):
#       start = time.time()
#
#       image = Image.open(image_path)
#       im_width, im_height = image.size
#
#       # rescale image if bigger than 640 x 640
#       if (im_width > 640 or im_height > 640):
#         image = rescale(im_width, im_height, image)
#
#       # the array based representation of the image will be used later in order to prepare the
#       # result image with boxes and labels on it.
#       image_np = load_image_into_numpy_array(image)
#
#       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#       image_np_expanded = np.expand_dims(image_np, axis=0)
#
#       # Actual detection.
#       output_dict = run_inference_for_single_image(image_np, detection_graph, sess)
#       total = time.time() - start
#
#       # Visualization of the results of a detection.
#       vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks'),
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=MAX_BOXES,
#         min_score_thresh=THRESHOLD,
#         line_thickness=2)
#
#       # Get the number of detections shown on image
#       count = len([i for i in output_dict['detection_scores'] if i >= THRESHOLD])
#
#       # Determine the ranking
#       rank_dict = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}
#
#       if count in range(0, 3):          # 0 - 2 objects [low]
#         rank = 1
#       elif count in range(3, 6):        # 3 - 5 objects [medium]
#         rank = 2
#       elif count in range(6, 9):        # 6 - 8 objects [high]
#         rank = 3
#       elif count >= 9:                  # 9 + objects [very high]
#         rank = 4
#
#       # display the image
#       plt.figure(figsize=IMAGE_SIZE)
#       plt.title("Detected: " + str(count) + "  |  Ranking: " + str(rank) + " [" + rank_dict[rank] + "]", fontsize=15)
#       plt.imshow(image_np)
#
#       print("[" + str(i) + "] Processed " + str(image_path) + " \t time = " + str(total))
