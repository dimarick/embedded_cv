#!/bin/bash

./embedded_cv $1 '-f mpegts -level:v high -profile:v high -b:v 3500k -framerate 15 -fflags nobuffer -flags low_delay -avioflags direct -fflags discardcorrupt -g 15 -threads 7' $2
