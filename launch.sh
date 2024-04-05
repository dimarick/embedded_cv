#!/bin/bash

export OPENCV_OPENCL_CACHE_CLEANUP=0
export OPENCV_OPENCL_FORCE=1

./embedded_cv $1 '-f flv -c:v libx264 -preset ultrafast -tune zerolatency -fflags nobuffer -avioflags direct -fflags discardcorrupt -g 15 -threads 7 -pix_fmt yuv422p' $2
