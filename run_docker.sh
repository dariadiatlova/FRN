#!/bin/bash

app=$PWD

docker build -t frn . && \
docker run -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v "$app":/app \
    frn