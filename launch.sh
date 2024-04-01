#!/bin/bash

./embedded_cv $1 '-f h264 -preset ultrafast -tune zerolatency -framerate 15 -fflags nobuffer -flags low_delay -avioflags direct -fflags discardcorrupt -g 15 -threads 7' $2

#ffplay -fflags nobuffer -flags low_delay -probesize 20000 -analyzeduration 1 -strict experimental -framedrop -fflags discardcorrupt udp://192.168.1.39:1234