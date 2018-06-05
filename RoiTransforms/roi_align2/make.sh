#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/


arch = sm_52

cd src/cuda/
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=arch
cd ../../
python build.py
cd ../../


#https://github.com/zhjpqq/pytorch-mask-rcnn
#cd roialign/roi_align/src/cuda/
#nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
#cd ../../
#python build.py
#cd ../../