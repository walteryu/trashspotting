# 03B - Toy detector script reference based on references below:
# Reference: https://github.com/walteryu/Deep-Learning/tree/master/tensorflow_toy_detector
# Article: https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95

# !echo '===> /content/models/research/object_detection/data/toy-detector/annotations/trainval.txt'
# !ls images | grep ".jpg" | sed s/.jpg// > ./annotations/trainval.txt
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/toy-detector/annotations'
# !ls -al /content/models/research/object_detection/data/toy-detector/annotations
# !echo ''
# !echo '===> cp /content/models/research/object_detection/data/toy-detector/annotations/xmls'
# !mkdir /content/models/research/object_detection/data/toy-detector/annotations/xmls
# !cp -rf /content/models/research/object_detection/data/toy-detector/annotations/*.xml \
#   /content/models/research/object_detection/data/toy-detector/annotations/xmls
# !echo ''
# !echo '===> ls /content/models/research/object_detection/data/toy-detector/images'
# !ls -al /content/models/research/object_detection/data/toy-detector/images
# !echo ''
#
# # !./toy-detector/create_pet_tf_record --data_dir=./toy-detector/images --output_dir=/toy-detector
# # !python /content/models/research/object_detection/data/toy-detector/create_pet_tf_record.py \
# #   --label_map_path=/content/models/research/object_detection/data/toy-detector/toy_label_map.pbtxt \
# #   --data_dir=. --output_dir=. --num_shards=1
# !python /content/models/research/object_detection/data/toy-detector/create_pet_tf_record.py \
#   --label_map_path=/content/models/research/object_detection/data/toy-detector/toy_label_map.pbtxt \
#   --data_dir=/content/models/research/object_detection/data/ \
#   --output_dir=/content/models/research/object_detection/data/toy-detector \
#   --include_masks=True
#
# # !echo '===> cd /content/models/research/object_detection/data/toy-detector'
# # %cd /content/models/research/object_detection/data/toy-detector
# # !echo ''
# # !echo '===> ls /content/models/research/object_detection/data/toy-detector'
# # !ls -al /content/models/research/object_detection/data/toy-detector
# # !echo ''
#
# # label_map.pbtxt path:
# # /content/models/research/object_detection/data/trashspotting/label_map.pbtxt
#
# # !echo '===> import google drive packages'
# # import os
# # from zipfile import ZipFile
# # from shutil import copy
# # from pydrive.auth import GoogleAuth
# # from pydrive.drive import GoogleDrive
# # from google.colab import auth
# # from oauth2client.client import GoogleCredentials
# # auth.authenticate_user()
# # gauth = GoogleAuth()
# # gauth.credentials = GoogleCredentials.get_application_default()
# # drive = GoogleDrive(gauth)
# # !echo ''
# #
# # !echo '===> cd /content/models/research/object_detection/data'
# # %cd /content/models/research/object_detection/data
# # !echo ''
# # !echo '===> ls /content/models/research/object_detection/data'
# # !ls -al /content/models/research/object_detection/data
# # !echo ''
# #
# # !echo '===> /content/models/research/object_detection/data/trash_dataset'
# # # Cop fileId from google drive shareable link
# # fileId = '1bPb7FTQ1yap_PgtzF5b_OuzVPWsIFLH4'
# # fileName = fileId + '.zip'
# # downloaded = drive.CreateFile({'id': fileId})
# # downloaded.GetContentFile(fileName)
# # ds = ZipFile(fileName)
# # ds.extractall()
# # os.remove(fileName)
# # print('Extracted zip file ' + fileName)
# # !echo '===> ls /content/models/research/object_detection/data'
# # !ls -al /content/models/research/object_detection/data
# # !echo ''
# #
# # !echo '===> cd /content/models/research/object_detection/data/trash_dataset'
# # %cd /content/models/research/object_detection/data/trash_dataset
# # !echo ''
# # !echo '===> ls /content/models/research/object_detection/data/trash_dataset'
# # !ls -al /content/models/research/object_detection/data/trash_dataset
# # !echo ''
