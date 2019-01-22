# reference: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
# !apt-get install protobuf-compiler python-pil python-lxml python-tk
# !pip install --user Cython
# !pip install --user contextlib2
# !pip install --user pillow
# !pip install --user lxml
# !protoc object_detection/protos/*.proto --python_out=.
# !export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# reference: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
# %cd ./object_detection/models/research
# !pwd
# !ls -al ./
# !wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# !wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
# !tar -xvf images.tar.gz
# !tar -xvf annotations.tar.gz
# !pwd
# !ls -al ./
# !ls -al ./object_detection
# !python object_detection/dataset_tools/create_pet_tf_record.py \
#     --label_map_path=object_detection/data/pet_label_map.pbtxt \
#     --data_dir=`pwd` \
#     --output_dir=`pwd`
# !pwd
# !ls -al ./

# reference: https://github.com/RomRoc/objdet_train_tensorflow_colab
# !git clone --quiet https://github.com/tensorflow/models.git
# !apt-get install -qq protobuf-compiler python-tk
# !pip install -q Cython contextlib2 pillow lxml matplotlib PyDrive
# !pip install -q pycocotools
# !ls -al ./models
# %cd ./object_detection/models/research
# !protoc object_detection/protos/*.proto --python_out=.
# import os
# os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
# !python object_detection/builders/model_builder_test.py
# !bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
# !python setup.py sdist
# !(cd slim && python setup.py sdist)
# %cd ./models/research
# !ls -al ./
# !python object_detection/builders/model_builder_test.py

# reference: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
# !git clone https://github.com/datitran/raccoon_dataset.git
# !python ./raccoon_dataset/generate_tfrecord.py
