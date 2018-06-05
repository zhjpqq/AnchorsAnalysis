#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py



# copy from !
#
# https://github.com/jwyang/faster-rcnn.pytorch


## according to bellow
## https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/make.sh