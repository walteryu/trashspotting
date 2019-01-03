# 06A - Colab Notebook Download
# Reference: https://github.com/bourdakos1/Custom-Object-Detection
# Tutorial: https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e

# Download training files
# from google.colab import files
# files.download('checkpoint')
# files.download('graph.pbtxt')
# files.download('model.ckpt-5.data-00000-of-00001')
# files.download('model.ckpt-5.index')
# files.download('model.ckpt-5.meta')

# Remaining training files but not necessary
# files.download('pipeline.config')
# files.download('test.record')
# files.download('train.record')

# Download training files; separate module after running model
# !echo '===> cd $PATH_TO_TRAIN'
# %cd $PATH_TO_TRAIN
# !echo ''
# !echo '===> ls $PATH_TO_TRAIN'
# !ls -al $PATH_TO_TRAIN
# !echo ''

# Relocate checkpoint files; do not run locally due to file size
# !mv ./checkpoint ../
# !mv ./graph.pbtxt ../
# !mv ./model.ckpt-5.data-00000-of-00001 ../
# !mv ./model.ckpt-5.index ../
# !mv ./model.ckpt-5.meta ../

# !echo '===> cd $PATH_TO_REPO'
# %cd $PATH_TO_REPO
# !echo ''
# !echo '===> ls $PATH_TO_REPO'
# !ls -al $PATH_TO_REPO
# !echo ''

# Export inference graph
# !python object_detection/export_inference_graph.py \
#         --input_type image_tensor \
#         --pipeline_config_path faster_rcnn_resnet101.config \
#         --trained_checkpoint_prefix model.ckpt-5 \
#         --output_directory output_inference_graph

# Test model!
# !python object_detection/object_detection_runner.py

# Download test results; needs to run in separate module to download
from google.colab import files
%cd $PATH_TO_REPO/output/test_images
files.download('image-1.jpg')
files.download('image-2.jpg')
files.download('image-3.jpg')
