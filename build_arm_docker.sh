ARCH=arm
PROJECT_ROOT=`pwd`

mkdir -p build/${ARCH}/OpenCV
mkdir -p build/${ARCH}/embedded_cv
mkdir -p release/${ARCH}

cd ${PROJECT_ROOT}/build/${ARCH}/OpenCV

PREFIX=${PROJECT_ROOT}/release/${ARCH}
OPEN_CV_PATH=${PROJECT_ROOT}/../opencv

CXXFLAGS="-mabi=lp64 -march=armv8.2-a+crypto+fp16+rcpc+dotprod -mcmodel=small -mcpu=cortex-a76.cortex-a55+crypto -mtune=cortex-a76.cortex-a55"

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
 PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
 PKG_CONFIG_SYSROOT_DIR=/ \
cmake \
  -DCMAKE_C_FLAGS="${CXXFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_INSTALL_PREFIX=${PREFIX}/OpenCV \
  -DWITH_OPENCL=ON \
  -DENABLE_VFPV3=OFF \
  -DENABLE_NEON=ON \
  -DBUILD_TBB=ON \
  -DBUILD_JAVA=OFF \
  -DBUILD_PACKAGE=ON \
  -DBUILD_PERF_TESTS=ON \
  -DBUILD_PROTOBUF=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TBB=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_opencv_highgui=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_java_bindings_generator=OFF \
  -DBUILD_opencv_objc_bindings_generator=ON \
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
  -DWITH_WEBP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=${OPEN_CV_PATH}/platforms/linux/aarch64-gnu.toolchain.cmake ${OPEN_CV_PATH} \
  && make -j 20 && make install || exit -1

cd ${PROJECT_ROOT}/build/${ARCH}/embedded_cv

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
 PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
 PKG_CONFIG_SYSROOT_DIR=/ \
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/embedded_cv \
  -DCMAKE_C_FLAGS="${CXXFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_PREFIX_PATH=${PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=${OPEN_CV_PATH}/platforms/linux/aarch64-gnu.toolchain.cmake ${PROJECT_ROOT} \
  && make -j 20 && make install
