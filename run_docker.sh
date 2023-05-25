#!/bin/bash

app=$PWD

docker build -t frn1 . && \
docker run -it --rm \
    --net=host --ipc=host \
    --gpus "all" \
    -v "$app":/app \
    frn1