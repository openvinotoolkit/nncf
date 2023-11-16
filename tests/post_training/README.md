# Post-training Quantization Conformance Suite

This is the test suite that takes PyTorch Timm models and runs post-training quantization on ImageNet dataset for the following three representations:

- PyTorch
- ONNX
- OpenVINO

The outcome of each quantization step is accuracy and performance with OpenVINO. The source representation is converted to OpenVINO IR at this step.

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

Once the environment is installed use the following command to run the test:

```bash
NUM_VAL_THREADS=8 pytest --data=<path_to_datasets> --output=./tmp tests/post_training/test_quantize_conformance.py
```

`NUM_VAL_THREADS` environment variable controls the number of parallel streams when validating the model.

Additional arguments:

- `--no-eval` to skip validation step
- `--fp32` to run validation of not quantized model
- `--cuda` to enable CUDA_TORCH backend
- `--subset-size=N` to force subset_size of calibration dataset
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
