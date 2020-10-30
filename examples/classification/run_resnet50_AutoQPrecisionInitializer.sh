#!/usr/bin/env bash

export PYTHONPATH=/path/to/autox

python main.py \
    -m test \
    --gpu-id 0 \
    --config configs/quantization/resnet50_imagenet_mixed_int_AutoQ_0.15.json \
    --data /data/dataset/imagenet/ilsvrc2012/torchvision \
    --workers 6 \
    --log-dir ./resnet50_AutoQPrecisionInitializer_run

