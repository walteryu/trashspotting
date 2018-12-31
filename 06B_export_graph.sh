# 06B - Chess Detection Algorithm (Export Graph)
# Reference: https://github.com/wagonhelm/TF_ObjectDetection_API
# Article: https://www.oreilly.com/ideas/object-detection-with-tensorflow

# # Copy object detection utilities to root directory
# PATH_TO_UTILS = '/content/models/research/object_detection/utils'
# !echo '===> ls $PATH_TO_UTILS'
# !ls -al $PATH_TO_UTILS
# !echo ''
#
# !echo '===> cp $PATH_TO_UTILS'
# %cd $PATH_TO_API
# # !cp -R /content/models/research/object_detection/utils/ ./
# !cp -R $PATH_TO_UTILS/. utils
# !echo ''

# Export inference graph
!echo '===> cd $PATH_TO_DATA'
%cd $PATH_TO_DATA
!echo ''
!echo '===> ls $PATH_TO_DATA'
!ls -al $PATH_TO_DATA
!echo ''

# Original script
# !rm -rf object_detection_graph
# !python /content/models/research/object_detection/export_inference_graph.py \
#     --input_type image_tensor \
#     --pipeline_config_path ./ssd_mobilenet_v1_pets.config \
#     --trained_checkpoint_prefix ./model.ckpt-0 \
#     --output_directory ./object_detection_graph
!python /content/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ./ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix ./model.ckpt-5 \
    --output_directory ./object_detection_graph
