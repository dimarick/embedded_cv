#!/bin/bash

./embedded_cv $1 | ffmpeg -f rawvideo -pixel_format bgr24 -video_size 640x480 -re  -i - -f h264 -preset ultrafast -tune zerolatency  -framerate 15 -fflags nobuffer -flags low_delay -avioflags direct -fflags discardcorrupt -g 10 $2