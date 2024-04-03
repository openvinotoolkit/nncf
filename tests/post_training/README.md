# Post-training Compression Conformance Suite

This is the test suite that takes PyTorch Timm or HuggingFace models and runs post-training compression on ImageNet or
some HuggingFace datasets for the following three representations:

- PyTorch
- ONNX
- OpenVINO

The outcome of each compression test case is accuracy and performance with OpenVINO.
The source representation is converted to OpenVINO IR at this step.

Test supports 2 different types of compression:

- Post-training quantization of weights and activations.
- Weight compression.

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

## Imagenet

<data>/imagenet/val - name of path
Since Torchvision `ImageFolder` class is used to work with data the ImageNet validation dataset should be structured accordingly. Below is an example of the `val` folder:

```text
n01440764
n01695060
n01843383
...
```

## Usage

Once the environment is installed use the following command to run all tests, including post-training quantization
and weight compression:

```bash
NUM_VAL_THREADS=8 pytest --data=<path_to_datasets> --output=./tmp tests/post_training/test_quantize_conformance.py
```

It's possible to run a suite of tests for the specific compression algorithm only.
For that append `::test_weight_compression` or `::test_ptq_quantization` to the `tests/post_training/test_quantize_conformance.py`.
For instance:

```bash
NUM_VAL_THREADS=8 pytest --data=<path_to_datasets> --output=./tmp tests/post_training/test_quantize_conformance.py::test_weight_compression
```

`NUM_VAL_THREADS` environment variable controls the number of parallel streams when validating the model.

Additional arguments:

- `--no-eval` to skip validation step
- `--fp32` to run validation of not quantized model
- `--cuda` to enable CUDA_TORCH backend
- `--subset-size=N` to force subset_size of calibration dataset
- `--batch-size=N` to use batch_size for calibration. Some of the models do not support --batch-size > 1. For such models, please, use --batch-size=1.
- `--benchmark` to collect throughput statistics, add `FPS` column to result.csv
- `--extra-columns` to add additional columns to reports.csv:
  - `Stat. collection time` - time of statistic collection
  - `Bias correction time` - time of bias correction
  - `Validation time` - time of validation

### Examples

Run for only OV backend:

```bash
pytest --data=<path_to_datasets> -k backend_OV tests/post_training/test_quantize_conformance.py
```

Run for only one model:

```bash
pytest --data=<path_to_datasets> -k timm/crossvit_9_240 tests/post_training/test_quantize_conformance.py
```

Run for only one model for OV backend:

```bash
pytest --data=<path_to_datasets> -k timm/crossvit_9_240_backend_OV tests/post_training/test_quantize_conformance.py
```

Only dump models:

```bash
pytest --data=<path_to_datasets> --no-eval tests/post_training/test_quantize_conformance.py
```

Fast dump models with `subset_size=1` for all models:

```bash
pytest --data=<path_to_datasets> --no-eval --subset-size 1 tests/post_training/test_quantize_conformance.py
```

Run test with collection of throughput statistics:

```bash
pytest --data=<path_to_datasets> --benchmark tests/post_training/test_quantize_conformance.py
```

Fast collection of throughput statistics:

```bash
pytest --data=<path_to_datasets> --benchmark --no-eval --subset-size 1 tests/post_training/test_quantize_conformance.py
```

Run test with additional columns:

```bash
pytest --data=<path_to_datasets> --extra-columns tests/post_training/test_quantize_conformance.py
```

Run test with calibration dataset having batch-size=10 for all models:

```bash
pytest --data=<path_to_datasets> --batch-size 10 tests/post_training/test_quantize_conformance.py
```
