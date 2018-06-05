#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py


## compile roi_pooling
#cd ../../
#cd model/roi_pooling/src
#echo "Compiling roi pooling kernels by nvcc..."
#nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
#	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
#cd ../
#python build.py