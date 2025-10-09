ARCH=amd64
PROJECT_ROOT=`pwd`

BUILD_TYPE=Debug
#BUILD_TYPE=Release

mkdir -p build/${ARCH}/${BUILD_TYPE}/OpenCV
mkdir -p build/${ARCH}/${BUILD_TYPE}/embedded_cv
mkdir -p release/${ARCH}/${BUILD_TYPE}

cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/OpenCV

PREFIX=${PROJECT_ROOT}/release/${ARCH}
OPEN_CV_PATH=${PROJECT_ROOT}/../opencv

cmake \
  -DCMAKE_CXX_FLAGS="-DCV_OPENCL_RUN_VERBOSE ${CMAKE_CXX_FLAGS} -O3 -msse3 -mtune=native" \
  -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/OpenCV \
  -DWITH_OPENCL=ON \
  -DBUILD_TBB=ON \
  -DBUILD_JAVA=OFF \
  -DBUILD_PACKAGE=ON \
  -DBUILD_PERF_TESTS=ON \
  -DBUILD_PROTOBUF=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_opencv_highgui=ON \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_java_bindings_generator=OFF \
  -DBUILD_opencv_objc_bindings_generator=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python_bindings_generator=OFF \
  -DBUILD_opencv_python_tests=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_JPEG=ON \
  -DWITH_LAPACK=ON \
  -DWITH_TBB=ON \
  -DWITH_TIFF=OFF \
  -DWITH_TIMVX=OFF \
  -DWITH_UEYE=OFF \
  -DWITH_V4L=ON \
  -DWITH_WEBP=OFF \
  -DWITH_EIGEN=OFF \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  ${OPEN_CV_PATH} \
  && make -j 20 && make install || exit -1

cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/embedded_cv

cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/embedded_cv \
  -DCMAKE_PREFIX_PATH=${PREFIX}/${BUILD_TYPE} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DARCH=${ARCH} \
  ${PROJECT_ROOT} \
  && make -j 20 && make install || exit -1

cp ${PREFIX}/${BUILD_TYPE}/embedded_cv/bin/embedded_cv ${PROJECT_ROOT}/