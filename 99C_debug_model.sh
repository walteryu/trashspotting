# 99 - Debugging model here since GTF is so complicated...

# Looking for create_pet_tf_record file since colab hides it...
# !echo '===> cd /content'
# %cd /content
# !echo ''
# !echo 'git clone tensorflow/models'
# !git clone https://github.com/tensorflow/models.git
# !echo '===> grep -nr faster_rcnn_inception_v2_pets.config'
# !grep -nr 'faster_rcnn_inception_v2_pets.config'
# !echo ''

# Searching for model config file:
!echo '===> cd /content/models/research/object_detection/'
%cd /content/models/research/object_detection
!echo ''
!echo '===> ls /content/models/research/object_detection/'
!ls -al /content/models/research/object_detection
!echo ''
!echo '===> cd /content/models/research/object_detection/samples'
%cd /content/models/research/object_detection/samples
!echo ''
!echo '===> ls /content/models/research/object_detection/samples'
!ls -al /content/models/research/object_detection/samples
!echo ''
