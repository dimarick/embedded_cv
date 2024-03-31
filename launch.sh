#!/bin/bash

./embedded_cv $1 | ffmpeg -f rawvideo -pixel_format bgr24 -s 640x480 -re  -i - \
  -f mpegts -level:v high -profile:v high -b:v 4000k -framerate 15 -fflags nobuffer -flags low_delay -avioflags direct -fflags discardcorrupt -g 20 $2
