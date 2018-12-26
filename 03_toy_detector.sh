# 03 - Toy detector model based on references below:

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

# !echo '===> /content/models/research/object_detection/data/trashspotting'
# !rm -rf ./trashspotting
# !git clone https://github.com/walteryu/trashspotting.git
# # !echo ''
# !echo '===> cd /content/models/research/object_detection/data/trashspotting'
# %cd /content/models/research/object_detection/data/trashspotting
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/trashspotting'
# !ls -al /content/models/research/object_detection/data/trashspotting
# !echo ''

# Reference: https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95
!rm -rf ./toy-detector
!git clone https://github.com/walteryu/toy-detector.git

# !echo '===> /content/models/research/object_detection/data/label_map.pbtxt'
# !echo "item {\n id: 1\n name: 'trash'\n}" > label_map.pbtxt
# !echo ''

# Copy files to correct locations for create_pet_tf_record
!echo '===> cd /content/models/research/object_detection/data/toy-detector'
%cd /content/models/research/object_detection/data/toy-detector
!echo ''
!echo '===> ls /content/models/research/object_detection/data/toy-detector'
!ls -al /content/models/research/object_detection/data/toy-detector
!echo ''
!echo '===> /content/models/research/object_detection/data/toy-detector/annotations/trainval.txt'
!ls images | grep ".jpg" | sed s/.jpg// > ./annotations/trainval.txt
!echo ''
!echo '===> ls /content/models/research/object_detection/data/toy-detector/annotations'
!ls -al /content/models/research/object_detection/data/toy-detector/annotations
!echo ''
!echo '===> cp /content/models/research/object_detection/data/toy-detector/annotations/xmls'
!mkdir /content/models/research/object_detection/data/toy-detector/annotations/xmls
!cp -rf /content/models/research/object_detection/data/toy-detector/annotations/*.xml \
  /content/models/research/object_detection/data/toy-detector/annotations/xmls
!echo ''

!echo '===> ls /content/models/research/object_detection/data/toy-detector/images'
!ls -al /content/models/research/object_detection/data/toy-detector/images
!echo ''

# !./toy-detector/create_pet_tf_record --data_dir=./toy-detector/images --output_dir=/toy-detector
# !python /content/models/research/object_detection/data/toy-detector/create_pet_tf_record.py \
#   --label_map_path=/content/models/research/object_detection/data/toy-detector/toy_label_map.pbtxt \
#   --data_dir=. --output_dir=. --num_shards=1
!python /content/models/research/object_detection/data/toy-detector/create_pet_tf_record.py \
  --label_map_path=/content/models/research/object_detection/data/toy-detector/toy_label_map.pbtxt \
  --data_dir=/content/models/research/object_detection/data/ \
  --output_dir=/content/models/research/object_detection/data/toy-detector \
  --include_masks=True

# !echo '===> cd /content/models/research/object_detection/data/toy-detector'
# %cd /content/models/research/object_detection/data/toy-detector
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/toy-detector'
# !ls -al /content/models/research/object_detection/data/toy-detector
# !echo ''

# label_map.pbtxt path:
# /content/models/research/object_detection/data/trashspotting/label_map.pbtxt

# !echo '===> import google drive packages'
# import os
# from zipfile import ZipFile
# from shutil import copy
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# !echo ''
#
# !echo '===> cd /content/models/research/object_detection/data'
# %cd /content/models/research/object_detection/data
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data'
# !ls -al /content/models/research/object_detection/data
# !echo ''
#
# !echo '===> /content/models/research/object_detection/data/trash_dataset'
# # Cop fileId from google drive shareable link
# fileId = '1bPb7FTQ1yap_PgtzF5b_OuzVPWsIFLH4'
# fileName = fileId + '.zip'
# downloaded = drive.CreateFile({'id': fileId})
# downloaded.GetContentFile(fileName)
# ds = ZipFile(fileName)
# ds.extractall()
# os.remove(fileName)
# print('Extracted zip file ' + fileName)
# !echo '===> ls /content/models/research/object_detection/data'
# !ls -al /content/models/research/object_detection/data
# !echo ''
#
# !echo '===> cd /content/models/research/object_detection/data/trash_dataset'
# %cd /content/models/research/object_detection/data/trash_dataset
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/trash_dataset'
# !ls -al /content/models/research/object_detection/data/trash_dataset
# !echo ''
