FROM debian:bookworm

RUN dpkg --add-architecture arm64 && apt update
RUN apt install -y linux-libc-dev:arm64
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

RUN apt install -y automake patch perl git tclsh
