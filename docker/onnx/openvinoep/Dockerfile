FROM openvino/onnxruntime_ep_ubuntu18:latest
USER root

ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG http_proxy
ARG https_proxy
ARG no_proxy

RUN echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}"
RUN echo "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}"
RUN echo "http_proxy=${http_proxy}"
RUN echo "https_proxy=${https_proxy}"
RUN echo "no_proxy=${no_proxy}"

RUN apt update && apt install -y git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip wheel

WORKDIR /opt/nncf_ws
ADD nncf/experimental/onnx/requirements.txt onnx_requirements.txt
RUN python3 -m pip install -r onnx_requirements.txt

ADD examples/experimental/onnx/requirements.txt example_requirements.txt
RUN python3 -m pip install -r example_requirements.txt

ADD nncf ./nncf
ADD setup.py ./
ADD README.md ./

RUN python3 -m pip install -e .

RUN ln -s /usr/bin/python3 /usr/bin/python

USER onnxruntimedev
WORKDIR /home/onnxruntimedev
ADD examples/experimental/onnx ./examples
