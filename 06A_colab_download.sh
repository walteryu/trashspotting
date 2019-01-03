# 06A - Colab Notebook Download
# Reference: https://github.com/wagonhelm/TF_ObjectDetection_API
# Article: https://www.oreilly.com/ideas/object-detection-with-tensorflow

# Download training files; separate module after running model
!echo '===> cd $PATH_TO_DATA'
%cd $PATH_TO_DATA
!echo ''
!echo '===> ls $PATH_TO_DATA'
!ls -al $PATH_TO_DATA
!echo ''

# Download training files
from google.colab import files
files.download('checkpoint')
files.download('graph.pbtxt')
files.download('model.ckpt-5.data-00000-of-00001')
files.download('model.ckpt-5.index')
files.download('model.ckpt-5.meta')
files.download('pipeline.config')
files.download('test.record')
files.download('train.record')
