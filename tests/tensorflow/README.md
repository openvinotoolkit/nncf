# Testing NNCF in Tensorflow

## Introduction

In this folder, there are test files available to test if the nncf module is installed and works properly in your local or server environment. It will test NNCF module with mock datasets(`cifar10` for classification, or `coco2017` for detection & segmentation) and mock models.

Before testing make sure that symlinks from `tests/tensorflow/data` are correct. They may be corrupted if the repo was downloaded to Windows machine via git without `core.symlinks` parameter enabled.

**NOTICE** : Test checkpoint for `mobilenet_v3_small_imagenet_rb_sparsity_int8` is updated for TF2.8 version, expected accuracy also updated to `67.55%` to `67.59%`.

---

## pre-commit test

A generic way to run TF pre-commit tests is via `make`:

```bash
make install-tensorflow-test
make test-tensorflow
```

Another way is to run `pytest` explicitly:

```bash
pytest tests/common tests/tensorflow \
  --junitxml nncf-tests.xml
```

The tests results will be saved in `nncf-tests.xml`.

## nightly-test

- Below is a description of the parameters to be used when building.

```text
--ignore-unknown-dependency
                      ignore dependencies whose outcome is not known
--data=DATA-DIR       Path to test datasets
--sota-checkpoints-dir=SOTA_CHECKPOINTS_DIR
                      Path to checkpoints directory for sota accuracy test
--sota-data-dir=SOTA_DATA_DIR
                      Path to datasets directory for sota accuracy test
--metrics-dump-path=METRICS_DUMP_PATH
                      Path to directory to store metrics. Directory must be empty or should not exist.Metric keeps in PROJECT_ROOT/test_results/metrics_dump_timestamp if param   not specified
--ov-data-dir=OV_DATA_DIR
                      Path to datasets directory for OpenVINO accuracy test
--run-openvino-eval   To run eval models via OpenVINO
--run-weekly-tests    To run weekly tests
--models-dir=MODELS_DIR
                      Path to checkpoints directory for weekly tests
--run-install-tests   To run installation tests
```

### test_sanity_sample.py

In this file, you will **test the basic training and evalutation loop in NNCF**. The `generate_config_params` function will generate some test configs that will be tested, and it will be saved into `CONFIG_PARAMS`. One example in `CONFIG_PARAMS` is like: `('classification', '{nncf-dir}/tests/tensorflow/data/configs/resnet50_cifar10_magnitude_sparsity_int8.json', 'cifar10', 'tfrecord')`.

The functions `test_model_eval`, `test_model_train`, `test_trained_model_eval`, or other similar functions are the key functions in this file. It receives parameters from config which is generated as sample, and the variable `main` in this function will get main function which is defined in each task(e.g. for classification: `examples/tensorflow/classification/main.py`). Each function will test the model from checkpoint, or train the model with 1~2 epochs, or test the onnx exporting of the tf model.

### test_weekly.py

In this file, you will **optimize and train the pre-trained models in `GLOBAL_CONFIG` with each dataset, and test the trained model's metrics within the `tolerance` value and `expected_accuracy`**. The `tolerance` term is the term on how much error to allow for relative accuracy, with the default value of 0.5. For example, if the expected accuracy is 75 and the tolerance value is 0.5, then an accuracy between 74.5 and 75.5 is allowed for test. You should give `--run-weekly-tests` parameter to run the whole process. It will take a long time because it will train the certain models.
Example of the tfds dataset structure is like below:

```text
tfds
├── cifar10
│   └── cifar10
│       └── 3.0.2
│           └──{TRAIN_DATAS}
├── coco2017
│   └── coco
│       └── 2017
│           └── 1.1.0
│               └──{TRAIN_DATAS}
└── imagenet2012
    └── imagenet2012
        └── 5.1.0
            └──{TRAIN_DATAS}
```

And example of the command of the weekly test is like below:

```bash
pytest --junitxml nncf-tests.xml tests/tensorflow/test_weekly.py -s \
  --run-weekly-tests \
  --data {PATH_TO_TFDS_OR_TFRECORDS_DATA_PATH} \
  --models-dir {PATH_TO_PRETRAINED_MODELS_CKPT_PATH} \
  --metrics-dump-path ./weekly_test_dump
```

### test_sota_checkpoints.py

In this file, you can **test whether the trained models from weekly test match the expected performance**. You can see the configurations are written in `sota_checkpoints_eval.json`, which contains the tasks / datasets / topologies. In topologies, it contains model name as a key and various datas such as config file path, ckpt path, target performance based on metric_type, compression method or etc. OV test will extract the `IR` or `frozen graph` from each model and test the extraced graph's accuracy. You can run the test from OV extracted model or eval from tensorflow model as follow:

```bash
pytest test_sota_checkpoints.py -s \
  -m oveval \
  --sota-checkpoints-dir={SOTA_CKPT_DIR} \
  --run-openvino-eval \
  --ov-data-dir={OV_DATA_DIR} \
  --metrics-dump-path ./ov_test_dump
```

```bash
pytest test_sota_checkpoints.py -s \
  --sota-checkpoints-dir={SOTA_CKPT_DIR} \
  --sota-data-dir={SOTA_DATA_DIR} \
  --metrics-dump-path ./eval_test_dump \
```
