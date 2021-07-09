FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git && \
rm -rf /var/lib/apt/lists/*


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
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
        python3-pip && \
    cd /usr/bin && \
    ln -s python3.8 python && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT cd /home/nncf \
               && python setup.py install --torch \
               && pip3 install -r examples/torch/requirements.txt \
               && bash
