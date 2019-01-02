# 12 - Mask R-CNN Demo with Keras and GTF (Colab Notebook)
# Reference: https://github.com/Tony607/colab-mask-rcnn
# Article: https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/

# Verify GPU:
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Install pycocotools
!pip install Cython
!git clone https://github.com/waleedka/coco

!pip install -U setuptools
!pip install -U wheel
!make install -C coco/PythonAPI

# Clone repo
!git clone https://github.com/matterport/Mask_RCNN

import os
os.chdir('./Mask_RCNN')
!git checkout 555126ee899a144ceff09e90b5b2cf46c321200c
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Install remaining modules
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

%matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Clone repo
!rm -rf ./trashspotting
!git clone https://github.com/walteryu/trashspotting.git
TRASH_DIR = os.path.join(ROOT_DIR, "trashspotting", "cnn_images")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Configure model
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Troubleshoot kera conflict: https://github.com/matterport/Mask_RCNN/issues/694
!pip install 'keras==2.1.6' --force-reinstall

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# >>>>> START >>>>>
# Run object detection
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['person', 'bicycle', 'car', 'motorcycle',
#                'cup', 'chair', 'couch', 'bed', 'desk', 'table', 'bottle',
#                'dining table', 'toilet', 'tv', 'laptop', 'cell phone', 'light',
#                'table', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
#                'book', 'textile-other', 'banner', 'pillow', 'blanket', 'curtain',
#                'cloth', 'clothes', 'napkin', 'towel', 'mat', 'rug', 'furniture-other',
#                'metal', 'plastic', 'paper', 'cardboard', 'carpet']
# Original class
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush', # end original classes
               'plate', 'table', 'plastic', 'metal', 'paper', 'cardboard']

# Load a random image from the images folder
# Photo: http://www.dot.ca.gov/dist11/trash/pics.html
# !wget http://www.dot.ca.gov/dist11/trash/2016/DSCN3217.jpg -P ./images
# image = skimage.io.imread(os.path.join(IMAGE_DIR, 'DSCN3217.jpg'))

for i in range(1,7):
  file_names = next(os.walk(TRASH_DIR))[2]
  # image = skimage.io.imread(os.path.join(TRASH_DIR, random.choice(file_names)))
  image = skimage.io.imread(os.path.join(TRASH_DIR, 'dtc' + str(i) + '.jpg'))
  results = model.detect([image], verbose=1)
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

for i in range(1,10):
  file_names = next(os.walk(TRASH_DIR))[2]
  # image = skimage.io.imread(os.path.join(TRASH_DIR, random.choice(file_names)))
  image = skimage.io.imread(os.path.join(TRASH_DIR, 'pixabay' + str(i) + '.jpg'))
  results = model.detect([image], verbose=1)
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

for i in range(1,52):
  file_names = next(os.walk(TRASH_DIR))[2]
  # image = skimage.io.imread(os.path.join(TRASH_DIR, random.choice(file_names)))
  image = skimage.io.imread(os.path.join(TRASH_DIR, 'trash' + str(i) + '.jpg'))
  results = model.detect([image], verbose=1)
  r = results[0]
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
