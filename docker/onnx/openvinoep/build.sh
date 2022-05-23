#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
WORK_DIR="${SCRIPT_DIR}/../../../"

cd $WORK_DIR && echo "WORK_DIR=$PWD"

docker build -t onnx_ptq_experimental:latest \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    -f docker/onnx/openvinoep/Dockerfile \
    .
