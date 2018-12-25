# 02 - Install object_detection module:
# Reference: https://github.com/RomRoc/objdet_train_tensorflow_colab
# Instructions: Always base dir on root path and run entire script

# Manage all directories closely by absolute path from root:
# !echo '===> home dir contents (do not use):'
# !ls -al ~/
# !echo ''
# !echo '===> root dir contents (based absolute path):'
# !ls -al /
# !echo ''

# Manage all directories closely by absolute path from root:
!echo '===> colab starting dir, /content:'
!pwd
!echo ''
!echo '===> colab starting dir contents (/content dir):'
!ls -al ./
!echo ''
!echo '===> /content dir contents (same as starting dir):'
!ls -al /content
!echo ''

# Clone tensorflow/model repo into /content dir:
!echo '===> clone models into /content (same as starting dir):'
!git clone --quiet https://github.com/tensorflow/models.git
!echo ''
# Install packages into system dir:
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

# Manage all directories closely by absolute path from root:
!echo '===> Current dir:'
!pwd
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

!echo '===> cd /content/models/research/object_detection'
%cd /content/models/research/object_detection
!echo ''
!echo '===> ls /content/models/research/object_detection'
!ls -al /content/models/research/object_detection
!echo ''
!echo '===> cd /content/models/research/object_detection/data'
%cd /content/models/research/object_detection/data
!echo ''
!echo '===> ls /content/models/research/object_detection/data'
!ls -al /content/models/research/object_detection/data
!echo ''

# !echo '===> /content/models/research/object_detection/data/label_map.pbtxt'
# !echo "item {\n id: 1\n name: 'trash'\n}" > label_map.pbtxt
# !echo ''
# !echo '===> /content/models/research/object_detection/data/trashspotting'
!rm -rf ./trashspotting
!git clone https://github.com/walteryu/trashspotting.git
# !echo ''
!echo '===> cd /content/models/research/object_detection/data/trashspotting'
%cd /content/models/research/object_detection/data/trashspotting
!echo ''
!echo '===> ls /content/models/research/object_detection/data/trashspotting'
!ls -al /content/models/research/object_detection/data/trashspotting
!echo ''

# label_map.pbtxt path:
# /content/models/research/object_detection/data/trashspotting/label_map.pbtxt

!echo '===> import google drive packages'
import os
from zipfile import ZipFile
from shutil import copy
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
!echo ''

!echo '===> cd /content/models/research/object_detection/data'
%cd /content/models/research/object_detection/data
!echo ''
!echo '===> ls /content/models/research/object_detection/data'
!ls -al /content/models/research/object_detection/data
!echo ''

!echo '===> /content/models/research/object_detection/data/trash_dataset'
# Cop fileId from google drive shareable link
fileId = '1bPb7FTQ1yap_PgtzF5b_OuzVPWsIFLH4'
fileName = fileId + '.zip'
downloaded = drive.CreateFile({'id': fileId})
downloaded.GetContentFile(fileName)
ds = ZipFile(fileName)
ds.extractall()
os.remove(fileName)
print('Extracted zip file ' + fileName)
!echo '===> ls /content/models/research/object_detection/data'
!ls -al /content/models/research/object_detection/data
!echo ''

!echo '===> cd /content/models/research/object_detection/data/trash_dataset'
%cd /content/models/research/object_detection/data/trash_dataset
!echo ''
!echo '===> ls /content/models/research/object_detection/data/trash_dataset'
!ls -al /content/models/research/object_detection/data/trash_dataset
!echo ''

# !echo '===> /content/models/research/object_detection/data/trash_dataset/annotations/trainval.txt'
!echo '===> cp /content/models/research/object_detection/data/trashspotting/annotations'
# !cp -rf /content/models/research/object_detection/data/trashspotting/annotations /content/models/research/object_detection/data/trash_dataset
!cp -rf /content/models/research/object_detection/data/trashspotting/annotations /content/models/research/object_detection/dataset_tools
!echo ''
!echo '===> cp /content/models/research/object_detection/data/trashspotting/images'
# !cp -rf /content/models/research/object_detection/data/trashspotting/images /content/models/research/object_detection/data/trash_dataset
!cp -rf /content/models/research/object_detection/data/trashspotting/images /content/models/research/object_detection/dataset_tools
!echo ''
!echo '===> ls /content/models/research/object_detection/'
!ls -al /content/models/research/object_detection
!echo ''
# create_tf_record file path: ls -al /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !echo '===> cp /content/models/research/object_detection/data/trashspotting/images'
# !cp -rf /content/models/research/object_detection/data/trashspotting/images /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !cp -rf /content/models/research/object_detection/data/trashspotting/images /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !echo ''
# !echo '===> cp /content/models/research/object_detection/data/trashspotting/annotations'
# !cp -rf /content/models/research/object_detection/data/trashspotting/annotations /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !cp -rf /content/models/research/object_detection/data/trashspotting/annotations /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools'
# !ls -al /content/models/research/object_detection/data/trash_dataset/models/research/object_detection/dataset_tools
# !echo ''

image_files=os.listdir('./images')
im_files=[x.split('.')[0] for x in image_files]
with open('./annotations/trainval.txt', 'w') as text_file:
  for row in im_files:
    text_file.write(row + '\n')
!echo ''

!echo '===> ls /content/models/research/object_detection/data/trash_dataset/annotations'
!ls -al /content/models/research/object_detection/data/trash_dataset/annotations
!echo ''
!echo '===> cd /content/models/research/object_detection/data/trash_dataset/annotations'
%cd /content/models/research/object_detection/data/trash_dataset/annotations
!echo ''
!echo '===> ls /content/models/research/object_detection/data/trash_dataset/annotations'
!ls -al /content/models/research/object_detection/data/trash_dataset/annotations
!echo ''

!echo '/content/models/research/object_detection/data/trash_dataset/annotations/trimaps/xxxxx.png'
!mkdir trimaps
from PIL import Image
image = Image.new('RGB', (640, 480))
for filename in os.listdir('./'):
  filename = os.path.splitext(filename)[0]
  image.save('trimaps/' + filename + '.png')
!echo ''
!echo '===> ls /content/models/research/object_detection/data/trash_dataset/annotations/trimaps'
!ls -al /content/models/research/object_detection/data/trash_dataset/annotations/trimaps
!echo ''

!echo '===> cd /content/models/research/object_detection/data'
%cd /content/models/research/object_detection/data
!echo ''
!echo '===> ls /content/models/research/object_detection/data'
!ls -al /content/models/research/object_detection/data
!echo ''

# Looking for create_pet_tf_record file since colab hides it...
!echo '===> cd /content'
%cd /content
!echo ''
!echo '===> grep -nr create_pet_tf_record'
!grep -nr 'create_pet_tf_record'
!echo ''

# Manage all directories closely by absolute path from root:
# !echo '===> Current dir:'
# !pwd
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
# !echo '===> cd /content/models/research/object_detection'
# %cd /content/models/research/object_detection
# !echo ''
# !echo '===> ls /content/models/research/object_detection'
# !ls -al /content/models/research/object_detection
# !echo ''

# # Change into dataset_tools dir to run create_pet_tf_record.py due to file path length:
# Correct path: models/research/object_detection/dataset_tools
!echo '===> cd /content/models/research/object_detection/dataset_tools/'
%cd /content/models/research/object_detection/dataset_tools
!echo ''
!echo '===> ls /content/models/research/object_detection/dataset_tools/'
!ls -al /content/models/research/object_detection/dataset_tools
!echo ''
!echo 'pwd'
!pwd
!echo ''
!echo 'ls ./'
!ls -al ./
!echo ''

!echo '/content/models/research/object_detection/data/tf_train.record'
!echo '/content/models/research/object_detection/data/tf_val.record'
# /content/models/research/object_detection/data/trashspotting/label_map.pbtxt
!python create_pet_tf_record.py --label_map_path=/content/models/research/object_detection/data/trashspotting/label_map.pbtxt --data_dir=. --output_dir=. --num_shards=1
!mv pet_faces_train.record-00000-of-00001 tf_train.record
!mv pet_faces_val.record-00000-of-00001 tf_val.record
!echo ''
!echo '===> ls /content/models/research/object_detection/dataset_tools/'
!ls -al /content/models/research/object_detection/dataset_tools
!echo ''
!echo 'pwd'
!pwd
!echo ''
!echo 'ls ./'
!ls -al ./
!echo ''

# !echo '===> cd /content/models/research/object_detection/data'
# %cd /content/models/research/object_detection/data
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data'
# !ls -al /content/models/research/object_detection/data
# !echo ''
#
# # !echo '===> Import modules (rnn model)'
# # import os
# # import shutil
# # import glob
# # import urllib
# # # Reference to debug urllib error:
# # # https://stackoverflow.com/questions/39975367/attributeerror-module-urllib-has-no-attribute-urlopen/39975383
# # import urllib.request
# # import tarfile
# # !echo ''
# #
# # !echo '===> Set vars (rnn model)'
# # MODEL = 'faster_rcnn_inception_v2_coco_2018_01_28'
# # MODEL_FILE = MODEL + '.tar.gz'
# # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# # DEST_DIR = 'pretrained_model'
# # !echo ''
# #
# # !echo '===> Download model (rnn model)'
# # if not (os.path.exists(MODEL_FILE)):
# #   # opener = urllib.URLopener()
# #   opener = urllib.request.urlopen()
# #   opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# #
# # tar = tarfile.open(MODEL_FILE)
# # tar.extractall()
# # tar.close()
# #
# # os.remove(MODEL_FILE)
# # if (os.path.exists(DEST_DIR)):
# #   shutil.rmtree(DEST_DIR)
# # os.rename(MODEL, DEST_DIR)
# # !echo ''
