#!/bin/bash

app=$PWD

docker build -t frn-sweep . && \
docker run -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v "$app":/app \
    frn-sweep