ARCH=amd64
PROJECT_ROOT=`pwd`

mkdir -p build/${ARCH}/OpenCV
mkdir -p build/${ARCH}/embedded_cv
mkdir -p release/${ARCH}

cd ${PROJECT_ROOT}/build/${ARCH}/OpenCV

PREFIX=${PROJECT_ROOT}/release/${ARCH}
OPEN_CV_PATH=${PROJECT_ROOT}/../opencv
#
#cmake \
#  -DCMAKE_INSTALL_PREFIX=${PREFIX}/OpenCV \
#  -DWITH_OPENCL=ON \
#  -DBUILD_JAVA=OFF \
#  -DBUILD_PACKAGE=ON \
#  -DBUILD_PERF_TESTS=ON \
#  -DBUILD_PROTOBUF=ON \
#  -DBUILD_SHARED_LIBS=OFF \
#  -DBUILD_TBB=ON \
#  -DBUILD_TESTS=OFF \
#  -DBUILD_opencv_highgui=ON \
#  -DBUILD_opencv_java=OFF \
#  -DBUILD_opencv_java_bindings_generator=OFF \
#  -DBUILD_opencv_objc_bindings_generator=ON \
#  -DBUILD_opencv_python3=OFF \
#  -DBUILD_opencv_python_bindings_generator=OFF \
#  -DBUILD_opencv_python_tests=OFF \
#  -DWITH_FFMPEG=ON \
#  -DWITH_GSTREAMER=OFF \
#  -DWITH_JPEG=ON \
#  -DWITH_LAPACK=ON \
#  -DWITH_TBB=ON \
#  -DWITH_TIFF=OFF \
#  -DWITH_TIMVX=OFF \
#  -DWITH_UEYE=OFF \
#  -DWITH_V4L=ON \
#  -DWITH_WEBP=ON \
#  -DCMAKE_BUILD_TYPE=Release \
#  ${OPEN_CV_PATH} \
#  && make -j 20 && make install

cd ${PROJECT_ROOT}/build/${ARCH}/embedded_cv

cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/embedded_cv \
  -DCMAKE_PREFIX_PATH=${PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  ${PROJECT_ROOT} \
  && make -j 20 && make install

cp ${PROJECT_ROOT}/release/amd64/embedded_cv/bin/embedded_cv ${PROJECT_ROOT}/