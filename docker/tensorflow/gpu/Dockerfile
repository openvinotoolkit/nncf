FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
        apt-transport-https=2.0 \
        git=2.25 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential=12.8 \
            libgl1-mesa-glx=21.2 \
            libglib2.0-dev=2.64 \
            wget=1.20 \
            curl=7.68 \
            zip=3.0 \
            unzip=6.0 \
            nano=4.8 \
            openssh-server=1:8.2 \
            openssh-client=1:8.2 \
            sudo=1.8 \
            python3=3.8 \
            python3-dev=3.8 \
            python3-pip=20.0 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/bin
RUN ln -s python3.8 python

RUN pip3 install --no-cache-dir --upgrade pip==23.3 \
        && pip3 install --no-cache-dir setuptools==69.0

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENTRYPOINT ["cd /home/nncf \
               && python setup.py install --tf \
               && pip3 install -r examples/tensorflow/requirements.txt \
               && bash"]
