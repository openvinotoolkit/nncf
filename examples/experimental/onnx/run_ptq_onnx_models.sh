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
    model_name=${config%.*}
    echo $model_name

    # Post-training quantization
    python $SCRIPT_DIR/run_ptq.py       \
        -c $CONFIGS_DIR/$config         \
        -m $MODEL_DIR/$model_name.onnx  \
        -o $OUTPUT_DIR                  \
        -ss $NUMBER_OF_SAMPLES

    # Accuracy check for the original model
    accuracy_check  \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -m $MODEL_DIR/$model_name.onnx              \
        --csv_result $OUTPUT_DIR/original_accuracy.csv

    # Accuracy check for the quantized model
    accuracy_check  \
        -c $CONFIGS_DIR/$config                     \
        -ss $NUMBER_OF_SAMPLES                      \
        -m $OUTPUT_DIR/$model_name-quantized.onnx   \
        --csv_result $OUTPUT_DIR/quantize_accuracy.csv

    # Benchmark the original model
    mkdir -p $OUTPUT_DIR/$model_name/original

    benchmark_app -m $MODEL_DIR/$model_name.onnx        \
        -report_type no_counters                        \
        -report_folder $OUTPUT_DIR/$model_name/original

    # Benchmark the quantized model
    mkdir -p $OUTPUT_DIR/$model_name/quantized

    benchmark_app -m $OUTPUT_DIR/$model_name-quantized.onnx \
        -report_type no_counters                            \
        -report_folder $OUTPUT_DIR/$model_name/quantized
done
