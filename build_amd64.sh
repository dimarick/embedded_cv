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
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O3 -msse3 -mtune=native" \
  -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/OpenCV \
  -DWITH_OPENCL=ON \
  -DWITH_OPENCL_SVM=ON \
  -DENABLE_VFPV3=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_PACKAGE=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_PROTOBUF=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TBB=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_opencv_3d=ON \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_calib3d=ON \
  -DBUILD_opencv_calib=ON \
  -DBUILD_opencv_core=ON \
  -DBUILD_opencv_dnn=OFF \
  -DBUILD_opencv_features2d=ON \
  -DBUILD_opencv_features=ON \
  -DBUILD_opencv_flann=ON \
  -DBUILD_opencv_gapi=OFF \
  -DBUILD_opencv_highgui=ON \
  -DBUILD_opencv_imgcodecs=ON \
  -DBUILD_opencv_imgproc=ON \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_java_bindings_generator=OFF \
  -DBUILD_opencv_js=OFF \
  -DBUILD_opencv_js_bindings_generator=OFF \
  -DBUILD_opencv_ml=OFF \
  -DBUILD_opencv_objc_bindings_generator=OFF \
  -DBUILD_opencv_objdetect=OFF \
  -DBUILD_opencv_photo=ON \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python_bindings_generator=OFF \
  -DBUILD_opencv_python_tests=OFF \
  -DBUILD_opencv_shape=OFF \
  -DBUILD_opencv_stereo=ON \
  -DBUILD_opencv_stitching=OFF \
  -DBUILD_opencv_superres=OFF \
  -DBUILD_opencv_ts=OFF \
  -DBUILD_opencv_video=ON \
  -DBUILD_opencv_videoio=ON \
  -DBUILD_opencv_videostab=OFF \
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
  -DWITH_1394=OFF \
  -DWITH_ADE=OFF \
  -DWITH_ARAVIS=OFF \
  -DWITH_AVIF=OFF \
  -DWITH_CANN=OFF \
  -DWITH_CLP=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_EIGEN=ON \
  -DWITH_FFMPEG=ON \
  -DWITH_FLATBUFFERS=OFF \
  -DWITH_FRAMEBUFFER=OFF \
  -DWITH_FRAMEBUFFER_XVFB=OFF \
  -DWITH_FREETYPE=OFF \
  -DWITH_GDAL=OFF \
  -DWITH_GDCM=OFF \
  -DWITH_GIGEAPI=OFF \
  -DWITH_GPHOTO2=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_GSTREAMER_0_10=OFF \
  -DWITH_GTK=ON \
  -DWITH_GTK_2_X=OFF \
  -DWITH_HALIDE=OFF \
  -DWITH_HPX=OFF \
  -DWITH_IMGCODEC_GIF=OFF \
  -DWITH_IMGCODEC_HDR=OFF \
  -DWITH_IMGCODEC_PFM=OFF \
  -DWITH_IMGCODEC_PXM=OFF \
  -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IPP=OFF \
  -DWITH_ITT=ON \
  -DWITH_JASPER=OFF \
  -DWITH_JPEG=ON \
  -DWITH_JPEGXL=OFF \
  -DWITH_LAPACK=ON \
  -DWITH_LIBREALSENSE=OFF \
  -DWITH_LIBV4L=OFF \
  -DWITH_MFX=OFF \
  -DWITH_OAK=OFF \
  -DWITH_OBSENSOR=OFF \
  -DWITH_ONNX=OFF \
  -DWITH_OPENCL=ON \
  -DWITH_OPENCLAMDBLAS=ON \
  -DWITH_OPENCLAMDFFT=ON \
  -DWITH_OPENCL_SVM=ON \
  -DWITH_OPENEXR=ON \
  -DWITH_OPENGL=OFF \
  -DWITH_OPENJPEG=ON \
  -DWITH_OPENMP=ON \
  -DWITH_OPENNI=OFF \
  -DWITH_OPENNI2=OFF \
  -DWITH_OPENVINO=OFF \
  -DWITH_OPENVX=OFF \
  -DWITH_PLAIDML=OFF \
  -DWITH_PNG=ON \
  -DWITH_PROTOBUF=OFF \
  -DWITH_PTHREADS_PF=ON \
  -DWITH_PVAPI=OFF \
  -DWITH_QT=OFF \
  -DWITH_QUIRC=OFF \
  -DWITH_SPNG=OFF \
  -DWITH_TBB=ON \
  -DWITH_TIFF=OFF \
  -DWITH_TIMVX=OFF \
  -DWITH_UEYE=OFF \
  -DWITH_UNICAP=OFF \
  -DWITH_UNIFONT=OFF \
  -DWITH_V4L=ON \
  -DWITH_VA=OFF \
  -DWITH_VA_INTEL=OFF \
  -DWITH_VTK=OFF \
  -DWITH_VULKAN=OFF \
  -DWITH_WAYLAND=OFF \
  -DWITH_WEBNN=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_XIMEA=OFF \
  -DWITH_XINE=OFF \
  -DWITH_ZLIB_NG=OFF \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  ${OPEN_CV_PATH} \
  && make -j 20 && make install || exit -1

cd ${PROJECT_ROOT}/build/${ARCH}/${BUILD_TYPE}/embedded_cv

cmake -GNinja -DCMAKE_INSTALL_PREFIX=${PREFIX}/${BUILD_TYPE}/embedded_cv \
  -DCMAKE_PREFIX_PATH=${PREFIX}/${BUILD_TYPE} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DARCH=${ARCH} \
  ${PROJECT_ROOT} \
  && ninja -j 20 && ninja install || exit -1

cp ${PREFIX}/${BUILD_TYPE}/embedded_cv/bin/embedded_cv ${PROJECT_ROOT}/