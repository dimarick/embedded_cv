export ARCH=arm
export PROJECT_ROOT=`pwd`
export BUILD_TYPE=Debug
#export BUILD_TYPE=Release

mkdir -p build/${ARCH}/${BUILD_TYPE}/hw_ctl
mkdir -p release/${ARCH}/${BUILD_TYPE}

export PREFIX=${PROJECT_ROOT}/release/${ARCH}
export OPEN_CV_PATH=${PROJECT_ROOT}/../../opencv

docker run \
    --volume "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --volume "${OPEN_CV_PATH}:${OPEN_CV_PATH}" \
    -e ARCH=${ARCH} \
    -e PROJECT_ROOT=${PROJECT_ROOT} \
    -e BUILD_TYPE=${BUILD_TYPE} \
    -e PREFIX=${PREFIX} \
    -e OPEN_CV_PATH=${OPEN_CV_PATH} \
    -w ${PROJECT_ROOT} \
    -it debian-build bash build_arm_docker.sh || exit -1

rsync -avc --progress ${PROJECT_ROOT}/release/${ARCH}/${BUILD_TYPE}/hw_ctl/bin/hw_ctl dima@192.168.1.41: || exit -1