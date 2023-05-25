FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV NCCL_SOCKET_IFNAME "lo"
ENV KMP_DUPLICATE_LIB_OK True

ENV NV_CUDNN_PACKAGE "$NV_CUDNN_PACKAGE_NAME=$NV_CUDNN_VERSION-1+cuda11.8"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc  \
    libsndfile1 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update -y \
    && apt-get install -y python3-pip \
    && apt-get install -y git-all \
    && apt-get install -y wget unzip p7zip-full libglib2.0-0 ffmpeg

RUN echo 'alias python=python3' >> ~/.bashrc

WORKDIR /app
COPY requirements.txt requirements.txt

# Activate conda environment for bash
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install ninja triton xformers==0.0.16
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]