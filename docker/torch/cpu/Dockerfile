FROM ubuntu:20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https=2.0 \
        git=2.25 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
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

ENTRYPOINT ["cd /home/nncf \
               && python setup.py install --torch \
               && pip3 install -r examples/torch/requirements.txt \
               && bash"]
