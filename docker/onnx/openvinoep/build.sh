#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
WORK_DIR="${SCRIPT_DIR}/../../../"

cd $WORK_DIR && echo "WORK_DIR=$PWD"

# TODO: Replace this image with Dockerfile.dev
docker build -t onnx_ptq_experimental:latest                \
    --build-arg http_proxy=$http_proxy                      \
    --build-arg https_proxy=$https_proxy                    \
    --build-arg no_proxy=$no_proxy                          \
    --build-arg PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL    \
    --build-arg PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST          \
    -f docker/onnx/openvinoep/Dockerfile                    \
    .

docker build -t onnx_ptq_experimental:dev                   \
    --build-arg http_proxy=$http_proxy                      \
    --build-arg https_proxy=$https_proxy                    \
    --build-arg no_proxy=$no_proxy                          \
    --build-arg PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL    \
    --build-arg PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST          \
    -f docker/onnx/openvinoep/Dockerfile.dev                \
    .
