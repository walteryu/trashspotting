# 02 - Install object_detection module:
# reference: https://github.com/RomRoc/objdet_train_tensorflow_colab

# !git clone --quiet https://github.com/tensorflow/models.git
# !apt-get install -qq protobuf-compiler python-tk
# !pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
# !pip install -q pycocotools

# !ls -al ./
# !ls -al ./models
# %cd ./object_detection/models/research

# !protoc object_detection/protos/*.proto --python_out=.
# import os
# os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
# !python object_detection/builders/model_builder_test.py

# download and train dataset
# %cd ./models/object_detection/datalab

# !echo "item {\n id: 1\n name: 'trash'\n}" > label_map.pbtxt
#
# # fileId = '1fBVMX66SlvrYa0oIau1lxt1_Vy-TdmWG'
# fileId = '1LF7ZP_DbfyMMFOnKhGDnAtpIMAwu4C7C'
#
# import os
# from zipfile import ZipFile
# from shutil import copy
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
#
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
#
# fileName = fileId + '.zip'
# downloaded = drive.CreateFile({'id': fileId})
# downloaded.GetContentFile(fileName)
# ds = ZipFile(fileName)
# ds.extractall()
# os.remove(fileName)
# print('Extracted zip file ' + fileName)
#
# image_files=os.listdir('images')
# im_files=[x.split('.')[0] for x in image_files]
# with open('annotations/trainval.txt', 'w') as text_file:
#   for row in im_files:
#     text_file.write(row + '\n')

!pwd
!ls -al ./models/research/object_detection
