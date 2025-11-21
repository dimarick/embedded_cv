FROM debian:trixie

RUN dpkg --add-architecture arm64 && apt update
RUN apt install -y linux-libc-dev:arm64 ocl-icd-opencl-dev:arm64 opencl-headers:arm64
#
#RUN apt install -y libavcodec-dev:arm64 \
# libavformat-dev:arm64 \
# libavutil-dev:arm64 \
# libswscale-dev:arm64 \
# libfreetype-dev:arm64 \
# libharfbuzz-dev:arm64

RUN apt install -y git \
 cmake \
 pkgconf \
 build-essential \
 ninja-build \
 crossbuild-essential-arm64 \
 unzip

RUN apt install -y git \
    ocl-icd-dev:arm64 \
    ocl-icd-opencl-dev:arm64 \
    libeigen3-dev:arm64


RUN apt install -y libpthreadpool-dev


RUN apt install -y automake patch perl git tclsh python3
RUN apt install -y zlib1g-dev:arm64 ocl-icd-libopencl1:arm64 mesa-opencl-icd:arm64
