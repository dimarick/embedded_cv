#!/bin/bash

apt update && \
apt upgrade && \
apt remove -y libwayland-client0 libwayland-server0 libx11-xcb1 && \
apt install -y v4l-utils ffmpeg netcat-openbsd ocl-icd-opencl ocl-icd-opencl-dev moreutils

cd /usr/lib && wget https://github.com/tsukumijima/libmali-rockchip/raw/master/lib/aarch64-linux-gnu/libmali-valhall-g610-g13p0-gbm.so
cd /lib/firmware/ && wget https://github.com/YangMame/mali-g610-firmware/raw/main/g18p0/mali_csffw.bin
