FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            apt-transport-https \
            git \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential \
            libgl1-mesa-glx \
            libglib2.0-dev \
            wget \
            curl \
            zip \
            unzip \
            nano \
            openssh-server \
            openssh-client \
            sudo \
            python3 \
            python3-dev \
            python3-pip \
        && cd /usr/bin \
        && ln -s python3.8 python \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
        && pip3 install setuptools

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ARG https_proxy
ARG no_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

ENTRYPOINT cd /home/nncf \
               && python setup.py install --tf \
               && pip3 install -r examples/tensorflow/requirements.txt \
               && bash
