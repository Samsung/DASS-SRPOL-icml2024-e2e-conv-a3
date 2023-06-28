#!/bin/bash

device_id=$1
docker run --rm -it --gpus "device=$device_id" \
    -v `pwd`:/Anomaly-analysis/ \
    anomaly_analysis:latest \
    bash -c "cd /Anomaly-analysis ; bash"
