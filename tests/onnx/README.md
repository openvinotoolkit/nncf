# Test for ONNX framework guide

We provide two types of tests.

1. Pre-commit test (no pytest markers)

    This is a test that the CI server runs for every PR. It consists of unit tests of ONNX features of NNCF. To run the pre-commit test, please execute the following command.

    ```bash
    pytest tests/onnx --junitxml nncf-tests.xml
    # (alias)
    make test-onnx
    ```

2. E2E test (pytest markers: `e2e_ptq` and `e2e_eval_reference_model`)

    This is a test to validate ONNX PTQ API functionality for the models in ONNX Model ZOO. It compares the quantized model accuracy with the references. To run the E2E test, please execute the following command.

    ```bash
    pytest tests/onnx -m e2e_ptq --model-dir (model_dir) --data-dir (data_dir) --output-dir (output_dir) --ckpt-dir (ckpt_dir) --anno-dir (anno_dir) --eval-size (eval_size) --ptq-size (ptq_size)
    ```

    You should give three arguments to run this test.

    1. `--model-dir`: The directory path which includes ONNX Model ZOO models (.onnx files). See [#prepare-models](benchmarking/README.md#benchmark-for-onnx-models-vision) for details.
    2. `--data-dir`: The directory path which includes datasets (ImageNet2012, COCO, Cityscapes, and VOC) [#prepare-dataset](benchmarking/README.md#1-prepare-dataset).
    3. `--output-dir`: The directory path where the test results will be saved.
    4. (Optional) `--model-names`: String containing model names to test. Model name is the prefix of the name of AccuracyChecker config before the '.' symbol. Please, provide the model names using ' ' as a separator.
    5. (Optional) `--ckpt-dir`: Directory path to save quantized models.
    6. (Optional) `--anno-dir`: Directory path for dataset annotations. Please refer to [OpenVINO accuracy checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker).
    7. (Optional) `--eval-size`: The number of samples for evaluation.
    8. (Optional) `--ptq-size`: The number of samples for calibrating quantization parameters.
    9. (Optional) `--enable-ov-ep`: If the parameter is set then the accuracy validation of the quantized models will be enabled for OpenVINOExecutionProvider.
    10. (Optional) `--disable-cpu-ep`: If the parameter is set then the accuracy validation of the quantized models will be disabled for CPUExecutionProvider.

    If you want to test the reference (not quantized) model accuracy - try the following command.

    ```bash
    pytest tests/onnx -m e2e_eval_reference_model \--model-dir (model_dir) --data-dir (data_dir) --output-dir (output_dir) --ckpt-dir (ckpt_dir) --anno-dir (anno_dir) --eval-size (eval_size) --ptq-size (ptq_size)
    ```
