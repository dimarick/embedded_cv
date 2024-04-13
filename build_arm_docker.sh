cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/OpenCV

CXXFLAGS="-mabi=lp64 -march=armv8.2-a+crypto+fp16+rcpc+dotprod -mcmodel=small -mcpu=cortex-a76.cortex-a55+crypto -mtune=cortex-a76.cortex-a55 -DCV_OPENCL_RUN_VERBOSE ${CMAKE_CXX_FLAGS}"

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
 PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
 PKG_CONFIG_SYSROOT_DIR=/ \
cmake \
  -DCMAKE_C_FLAGS="${CXXFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/OpenCV \
  -DWITH_OPENCL=ON \
  -DWITH_OPENCL_SVM=ON \
  -DENABLE_VFPV3=OFF \
  -DENABLE_NEON=ON \
  -DBUILD_TBB=ON \
  -DBUILD_JAVA=OFF \
  -DBUILD_PACKAGE=ON \
  -DBUILD_PERF_TESTS=ON \
  -DBUILD_PROTOBUF=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_opencv_highgui=OFF \
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
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_TOOLCHAIN_FILE=${OPEN_CV_PATH}/platforms/linux/aarch64-gnu.toolchain.cmake ${OPEN_CV_PATH} \
  && make -j 20 && make install || exit -1

cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/embedded_cv

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
 PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
 PKG_CONFIG_SYSROOT_DIR=/ \
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/embedded_cv \
  -DCMAKE_C_FLAGS="${CXXFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_PREFIX_PATH=${PREFIX}/${BUILD_TYPE} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DARCH=${ARCH} \
  -DCMAKE_TOOLCHAIN_FILE=${OPEN_CV_PATH}/platforms/linux/aarch64-gnu.toolchain.cmake ${PROJECT_ROOT} \
  && make -j 20 && make install
