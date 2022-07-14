# Test for ONNX framework guide

We provide two types of tests.

1. Pre-commit test (no pytest markers)

    This is a test that the CI server runs for every PR. It includes unit testing of ONNX framework-specific features of NNCF. In order to merge PR into the develop branch, all tests in it must be green. If a new feature or a bug fix is implemented, an appropriate unit test to validate those implementation should also be added to the pre-commit test. To run the pre-commit test, please execute the following command.

    ```bash
    $ pytest tests/onnx --junitxml nncf-tests.xml
    # (alias)
    $ make test-onnx
    ```

2. E2E test for ONNX Model ZOO (pytest markers: `e2e_ptq` and `e2e_eval_original_model`)

    This is a test to validate ONNX PTQ API functionality for the models in ONNX Model ZOO. There is no obligation for developers to check this test to merge their PR. It checks the quantized model accuracy and performance and compare them to the references. Our CI server runs it periodically, and if defects are found, the merged changes during the test cycle should be investigated. To run the E2E test, please execute the following command.

    ```bash
    $ pytest tests/onnx -m e2e_ptq --model-dir (model_dir) --data-dir (data_dir) --output-dir (output_dir) --eval-size (eval_size)
    ```

    You should give three arguments to run this test.

    1. model_dir: The directory path which includes ONNX Model ZOO models (.onnx files).
    2. data_dir: The directory path which includes datasets (ImageNet2012, COCO, Cityscapes, and VOC).
    3. output_dir: The directory path where the test results will be saved.

    (Optional) If you want to test the original (not quantized) model accuracy and performance, try the following command.

    ```bash
    $ pytest tests/onnx -m e2e_eval_original_model --model-dir (model_dir) --data-dir (data_dir) --output-dir (output_dir) --eval-size (eval_size)
    ```
