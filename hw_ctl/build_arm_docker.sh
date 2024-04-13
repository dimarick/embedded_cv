CXXFLAGS="-mabi=lp64 -march=armv8.2-a+crypto+fp16+rcpc+dotprod -mcmodel=small -mcpu=cortex-a76.cortex-a55+crypto -mtune=cortex-a76.cortex-a55 -DCV_OPENCL_RUN_VERBOSE ${CMAKE_CXX_FLAGS}"

cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/hw_ctl

PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig \
 PKG_CONFIG_LIBDIR=/usr/lib/aarch64-linux-gnu \
 PKG_CONFIG_SYSROOT_DIR=/ \
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/hw_ctl \
  -DCMAKE_C_FLAGS="${CXXFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_PREFIX_PATH=${PREFIX}/${BUILD_TYPE} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DARCH=${ARCH} \
  -DCMAKE_TOOLCHAIN_FILE=${OPEN_CV_PATH}/platforms/linux/aarch64-gnu.toolchain.cmake ${PROJECT_ROOT} \
  && make -j 20 && make install
