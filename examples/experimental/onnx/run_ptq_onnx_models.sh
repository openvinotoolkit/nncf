#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")

if [ $# -ne 5 ]; then
    echo "illegal number of parameters"
    echo "E.g. ./run_ptq_onnx_models.sh [classification|det_and_seg] <MODEL_DIR> <OUTPUT_DIR> <DATASET_DIR> <NUMBER_OF_SAMPLES>"
    exit 2
fi

case $1 in
    classification|det_and_seg)
        CONFIGS_DIR=${SCRIPT_DIR}/$1/onnx_models_configs;
        echo "CONFIGS_DIR=$CONFIGS_DIR"
        ;;
    *)
        echo "You should choose classification or det_and_seg. E.g. ./run_ptq_onnx_models.sh classification ..."
        exit 2
        ;;
esac

MODEL_DIR=$2
OUTPUT_DIR=$3
DATASET_DIR=$4
NUMBER_OF_SAMPLES=$5

echo "MODEL_DIR=$MODEL_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASET_DIR=$DATASET_DIR"
echo "NUMBER_OF_SAMPLES=$NUMBER_OF_SAMPLES"

for config in `ls $CONFIGS_DIR`; do
    model_name=${config%.*}

    echo "1) RUN PTQ for $model_name"
    # Post-training quantization
    python $SCRIPT_DIR/run_ptq.py               \
        -c $CONFIGS_DIR/$config                 \
        -m $MODEL_DIR/$model_name.onnx          \
        -o $OUTPUT_DIR                          \
        -d $SCRIPT_DIR/dataset_definitions.yml  \
        -s $DATASET_DIR                         \
        -a $DATASET_DIR                         \
        -ss $NUMBER_OF_SAMPLES

    echo "2) Check model accuracy for $model_name (original)"
    # Accuracy check for the original model
    python $SCRIPT_DIR/accuracy_checker.py  \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -m $MODEL_DIR/$model_name.onnx              \
        -d $SCRIPT_DIR/dataset_definitions.yml      \
        -s $DATASET_DIR                             \
        -a $DATASET_DIR                             \
        --csv_result $OUTPUT_DIR/original_accuracy.csv

    echo "3) Check model accuracy for $model_name (quantized)"
    # Accuracy check for the quantized model
    python $SCRIPT_DIR/accuracy_checker.py  \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -m $OUTPUT_DIR/$model_name-quantized.onnx   \
        -d $SCRIPT_DIR/dataset_definitions.yml      \
        -s $DATASET_DIR                             \
        -a $DATASET_DIR                             \
        --csv_result $OUTPUT_DIR/quantized_accuracy.csv

    echo "4) Benchmark model latency for $model_name (original)"
    # Benchmark the original model
    python $SCRIPT_DIR/performance_checker.py       \
        -m $MODEL_DIR/$model_name.onnx              \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -d $SCRIPT_DIR/dataset_definitions.yml      \
        -s $DATASET_DIR                             \
        -a $DATASET_DIR                             \
        -o $OUTPUT_DIR/original_performance.csv

    echo "5) Benchmark model latency for $model_name (quantized)"
    # Benchmark the quantized model
    python $SCRIPT_DIR/performance_checker.py       \
        -m $OUTPUT_DIR/$model_name-quantized.onnx   \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -d $SCRIPT_DIR/dataset_definitions.yml      \
        -s $DATASET_DIR                             \
        -a $DATASET_DIR                             \
        -o $OUTPUT_DIR/quantized_performance.csv

done
