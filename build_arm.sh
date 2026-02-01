#!/bin/bash

if [[ "$SSH_SERVER" == "" ]]; then
  export SSH_SERVER=dima@192.168.3.40
fi

export ARCH=arm
export PROJECT_ROOT=`pwd`
export BUILD_TYPE=Debug
#export BUILD_TYPE=Release

mkdir -p build/${ARCH}/${BUILD_TYPE}/OpenCV
mkdir -p build/${ARCH}/${BUILD_TYPE}/embedded_cv
mkdir -p release/${ARCH}/${BUILD_TYPE}

export PREFIX=${PROJECT_ROOT}/release/${ARCH}
export OPEN_CV_PATH=${PROJECT_ROOT}/../opencv

rsync -avc --progress ${PROJECT_ROOT}/DisparityEvaluator3ch.cl ${SSH_SERVER}: || exit -1

docker build -t debian-build . || exit -1
docker run \
    --volume '/home/dima/embedded_cv/:/home/dima/embedded_cv/' \
    --volume '/home/dima/opencv/:/home/dima/opencv/' \
    -e ARCH=${ARCH} \
    -e PROJECT_ROOT=${PROJECT_ROOT} \
    -e BUILD_TYPE=${BUILD_TYPE} \
    -e PREFIX=${PREFIX} \
    -e OPEN_CV_PATH=${OPEN_CV_PATH} \
    -w /home/dima/embedded_cv/ \
    -it debian-build bash build_arm_docker.sh || exit -1

rsync -avc --progress ${PROJECT_ROOT}/release/${ARCH}/${BUILD_TYPE}/embedded_cv/bin/* ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/release/${ARCH}/${BUILD_TYPE}/OpenCV/lib/*.so* ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/map0.bin ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/map1.bin ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/mapa.bin ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/launch.sh ${SSH_SERVER}: || exit -1
rsync -avc --progress ${PROJECT_ROOT}/ws_ctl/web/* --exclude=CMakeLists.txt ${SSH_SERVER}:web || exit -1