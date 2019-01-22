# 16 - Facebook Detectron Module (Mask R-CNNV)
# Article: https://hackernoon.com/how-to-use-detectron-facebooks-free-platform-for-object-detection-9d41e170bbcb
# Github: https://github.com/facebookresearch/Detectron

%cd /content
# !git clone --recursive https://github.com/pytorch/pytorch/
# %cd pytorch/caffe2
# !ls -alias

# !make
# !ls -alias

# %cd build
# !sudo make install
# !python -c 'from caffe2.python import core' 2>/dev/null
# !echo "Success" || echo "Failure"

# Reference: https://github.com/facebookarchive/caffe2/issues/1811
# !apt-add-repository ppa:facebook/caffe2
# !apt-get update
# !apt-get install caffe2-cu90

# Reference: https://tech.amikelive.com/node-706/comprehensive-guide-installing-caffe2-with-gpu-support-by-building-from-source-on-ubuntu-16-04/
%cd pytorch
!git submodule update --init
!mkdir build
%cd build
!cmake ..
!sudo make -j"$(nproc)" install

!sudo ldconfig
!sudo updatedb
!locate libcaffe2.so
!locate caffe2 | grep /usr/local/include/caffe2
