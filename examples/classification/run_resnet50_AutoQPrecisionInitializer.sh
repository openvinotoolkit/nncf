#!/usr/bin/env bash

export PYTHONPATH=~/workdir/nncf_pytorch:~/workdir/autox

python main.py \
    -m test \
    --gpu-id 0 \
    --config configs/quantization/resnet50_imagenet_mixed_int_AutoQ_0.15.json \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision \
    --workers 16 \
    --log-dir ./resnet50_AutoQPrecisionInitializer_run

