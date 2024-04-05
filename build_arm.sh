ARCH=arm
PROJECT_ROOT=`pwd`

BUILD_TYPE=Debug
#BUILD_TYPE=Release

mkdir -p build/${ARCH}/${BUILD_TYPE}/OpenCV
mkdir -p build/${ARCH}/embedded_cv
mkdir -p release/${ARCH}/${BUILD_TYPE}

PREFIX=${PROJECT_ROOT}/release/${ARCH}
OPEN_CV_PATH=${PROJECT_ROOT}/../opencv

docker build -t debian-build . || exit -1
docker run --volume '/home/dima/embedded_cv/:/home/dima/embedded_cv/' --volume '/home/dima/opencv/:/home/dima/opencv/' -w /home/dima/embedded_cv/ -it debian-build bash build_arm_docker.sh || exit -1

rsync -av --progress ${PROJECT_ROOT}/release/arm/embedded_cv/bin/embedded_cv dima@192.168.1.41: || exit -1
rsync -av --progress ${PROJECT_ROOT}/launch.sh dima@192.168.1.41: || exit -1