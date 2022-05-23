#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
CONFIGS_DIR=${SCRIPT_DIR}/classification/onnx_models_configs

MODEL_DIR=$1
OUTPUT_DIR=$2
NUMBER_OF_SAMPLES=$3

echo "MODEL_DIR=$MODEL_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUMBER_OF_SAMPLES=$NUMBER_OF_SAMPLES"

for config in `ls $CONFIGS_DIR`; do
    model_name=$(echo "$config" | cut -f 1 -d '.')
    echo $model_name
    python ${SCRIPT_DIR}/run_ptq.py     \
        -c ${CONFIGS_DIR}/$config -m    \
        ${MODEL_DIR}/${model_name}.onnx \
        -o ${OUTPUT_DIR} \
        -ss ${NUMBER_OF_SAMPLES}
done
