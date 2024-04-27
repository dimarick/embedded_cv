#!/bin/bash

BUILD_DIR=`pwd`

#sudo rm -r ../srs/trunk/objs/*
cd ../srs/trunk

PROJECT_DIR=`pwd`

./configure --static=on --h265=off --srt=off --https=off && make

rsync -avc --progress ${PROJECT_DIR}/objs/srs ${BUILD_DIR}/objs/ || exit -1
rsync -avc --progress ${PROJECT_DIR}/objs/nginx ${BUILD_DIR}/objs/ || exit -1
rsync -avc --progress ${PROJECT_DIR}/conf/rtmp2rtc.conf ${BUILD_DIR}/conf/ || exit -1