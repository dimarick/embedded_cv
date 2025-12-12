#!/bin/bash

export OPENCV_OPENCL_CACHE_CLEANUP=0
export OPENCV_OPENCL_FORCE=1
export LD_LIBRARY_PATH=`pwd`

./cv $1 $2
