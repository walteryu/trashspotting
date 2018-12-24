# 02 - Install object_detection module:
# reference: https://github.com/RomRoc/objdet_train_tensorflow_colab

# !git clone --quiet https://github.com/tensorflow/models.git
# !apt-get install -qq protobuf-compiler python-tk
# !pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
# !pip install -q pycocotools

# %cd ./models/research
# !protoc object_detection/protos/*.proto --python_out=.
# import os
# os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
# !python object_detection/builders/model_builder_test.py

# download and train dataset
# !pwd
# !ls -al ./object_detection/
# !ls -al ./object_detection/data

# %cd ./models/object_detection/data
# !pwd
# %cd ./object_detection/data
# !echo "item {\n id: 1\n name: 'trash'\n}" > label_map.pbtxt
# !ls -al ./object_detection/data
# !ls -al ./

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

# !pwd
# !ls -al ./
# %cd ./trash_dataset
# !pwd
#
# image_files=os.listdir('images')
# im_files=[x.split('.')[0] for x in image_files]
# with open('annotations/trainval.txt', 'w') as text_file:
#   for row in im_files:
#     text_file.write(row + '\n')
#
# !ls -al ./annotations

# !pwd
# %cd ./annotations
# !mkdir trimaps

# !pwd
# !ls -al ./
#
# from PIL import Image
# image = Image.new('RGB', (640, 480))
# for filename in os.listdir('./'):
#   filename = os.path.splitext(filename)[0]
#   image.save('trimaps/' + filename + '.png')

# !pwd
# %cd ./object_detection/data
# !pwd

!python ~/models/content/research/object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=label_map.pbtxt --data_dir=. --output_dir=. --num_shards=1
!mv pet_faces_train.record-00000-of-00001 tf_train.record
!mv pet_faces_val.record-00000-of-00001 tf_val.record
