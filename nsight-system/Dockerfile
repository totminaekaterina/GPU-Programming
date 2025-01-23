FROM nvidia/cuda:12.6.0-devel-ubuntu20.04

USER root


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


RUN apt-get update -y && \
    apt-get install -y \
        software-properties-common \
        build-essential \
        cmake \
        wget \
        curl \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libpng-dev \
        g++ \
        gcc \
        pkg-config \
        libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# .deb пакет Nsight Systems CLI
RUN wget https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/NsightSystems-linux-cli-public-2024.6.1.90-3490548.deb -O /tmp/nsight-systems.deb

RUN dpkg -i /tmp/nsight-systems.deb || apt-get install -f -y && \
    rm /tmp/nsight-systems.deb

# RUN ln -sf /opt/nvidia/nsight-systems-*/host/target-linux-x64/nsys /usr/local/bin/nsys

RUN nsys --version

WORKDIR /cuda

COPY Makefile /cuda/Makefile
COPY *.cu /cuda/

CMD ["make", "profile"]


# docker build --no-cache -t cuda-cpp-docker:latest .
# docker run -it --gpus all -v C:/Users/totmi/Desktop/CUDA:/cuda cuda-cpp-docker:latest bin/bash
# nsys --version