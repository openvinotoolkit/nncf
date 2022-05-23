FROM alpine/git AS git

ARG http_proxy
ARG https_proxy

RUN echo "http_proxy=${http_proxy}"
RUN echo "https_proxy=${https_proxy}"

WORKDIR /tmp
RUN git clone -b 2022.1.0 --depth 1 https://github.com/openvinotoolkit/open_model_zoo.git
RUN git clone -b 2022.1.0 --depth 1 https://github.com/openvinotoolkit/openvino.git

FROM openvino/onnxruntime_ep_ubuntu18:latest
USER root

WORKDIR /opt
COPY --from=git /tmp/open_model_zoo/tools/accuracy_checker ./accuracy_checker
COPY --from=git /tmp/openvino/tools/benchmark_tool ./benchmark_tool

ARG http_proxy
ARG https_proxy

RUN python3 -m pip install -U pip wheel
RUN python3 -m pip install benchmark_tool/
RUN python3 -m pip install accuracy_checker/

RUN apt update && apt install -y git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY --from=git /tmp/open_model_zoo/data/dataset_definitions.yml ./dataset_definitions.yml
RUN chown onnxruntimedev:onnxruntimedev dataset_definitions.yml

WORKDIR /opt/nncf_ws
ADD nncf/experimental/onnx/requirements.txt onnx_requirements.txt
RUN python3 -m pip install -r onnx_requirements.txt

ADD . ./
RUN python3 -m pip install -e .

USER onnxruntimedev
WORKDIR /home/onnxruntimedev
