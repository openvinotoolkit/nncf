ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:24.04

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London" \
    NODE_EXTRA_CA_CERTS="/usr/local/share/ca-certificates/IntelProxyRootCA-Base64.crt" \
    GIT_SSL_CAINFO="/usr/local/share/ca-certificates/IntelProxyRootCA-Base64.crt" \
    PIP_CERT="/usr/local/share/ca-certificates/IntelProxyRootCA-Base64.crt" \
    REQUESTS_CA_BUNDLE="/usr/local/share/ca-certificates/IntelProxyRootCA-Base64.crt" \
    SSL_CERT_FILE="/usr/local/share/ca-certificates/IntelProxyRootCA-Base64.crt"

RUN apt-get update && \
    apt-get --assume-yes install --no-install-recommends \
        git \
        git-lfs \
        ca-certificates \
        build-essential \
        # python3 \
        # python3-pip \
        # python3-venv \
        ninja-build \
        libgl1-mesa-dev \
        libglib2.0-0 \
        wget \
        make \
        libpython3.12-dev \
        pkg-config \
        libffi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run -O /tmp/cuda_installer.run && \
    sh /tmp/cuda_installer.run --toolkit --silent && \
    rm -f /tmp/cuda_installer.run
