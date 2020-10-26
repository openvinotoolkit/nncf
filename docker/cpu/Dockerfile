FROM ubuntu:18.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git  && \
rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        wget \
        gcc \
        curl \
        zip \
        unzip \
        nano \
        openssh-server \
        openssh-client \
        sudo \
        g++ \
    && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3-pip \
    virtualenv && \
    cd /usr/bin && \
    ln -s python3.6 python && \
    curl https://bootstrap.pypa.io/get-pip.py | python \
    && \
    rm -rf /var/lib/apt/lists/*


ARG https_proxy
ARG no_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

RUN pip install setuptools

ENTRYPOINT cd /home/nncf/ && python setup.py develop --cpu-only && bash
