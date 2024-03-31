ARCH=arm
PROJECT_ROOT=`pwd`

mkdir -p build/${ARCH}/OpenCV
mkdir -p build/${ARCH}/embedded_cv
mkdir -p release/${ARCH}

PREFIX=${PROJECT_ROOT}/release/${ARCH}
OPEN_CV_PATH=${PROJECT_ROOT}/../opencv

docker build -t debian-build . || exit -1
docker run --volume '/home/dima/embedded_cv/:/home/dima/embedded_cv/' --volume '/home/dima/opencv/:/home/dima/opencv/' -w /home/dima/embedded_cv/ -it debian-build bash build_arm_docker.sh || exit -1

scp ${PROJECT_ROOT}/release/arm/embedded_cv/bin/embedded_cv dima@192.168.1.41: || exit -1
scp ${PROJECT_ROOT}/launch.sh dima@192.168.1.41: || exit -1