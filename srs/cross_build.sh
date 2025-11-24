#!/bin/bash

BUILD_DIR=`pwd`

docker build -t debian-build . || exit -1
cd ../srs

PROJECT_DIR=`pwd`

cp ${BUILD_DIR}/srs/cross_build_docker.sh ${PROJECT_DIR}/cross_build_docker.sh

docker run \
    --volume "${PROJECT_DIR}/:${PROJECT_DIR}/" \
    -w ${PROJECT_DIR}/trunk/ \
    -it debian-build bash ${PROJECT_DIR}/cross_build_docker.sh

rsync -avc --progress ${PROJECT_DIR}/trunk/objs/srs dima@192.168.1.41:objs/ || exit -1
rsync -avc --progress ${PROJECT_DIR}/trunk/objs/nginx dima@192.168.1.41:objs/ || exit -1
rsync -avc --progress ${PROJECT_DIR}/trunk/conf/rtmp2rtc.conf dima@192.168.1.41:conf/ || exit -1
rsync -avc --progress ${BUILD_DIR}/srs/srs.sh dima@192.168.1.41: || exit -1