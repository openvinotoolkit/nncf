# Post-training Quantization Conformance Suite
This is the test suite that takes PyTorch Timm models and runs post-training quantization on ImageNet dataset for the following three representations:
- PyTorch
- ONNX
- OpenVINO

The outcome of each quantization step is accuracy and performance with OpenVINO. The source representation is converted to OpenVINO IR at this step.

## Installation
```
pip install -r requirements.txt
```

## Data preparation

### Imagenet

<data>/imagenet/val - name of path
Since Torchvision `ImageFolder` class is used to work with data the ImageNet validation dataset should be structured accordingly. Below is an example of the `val` folder:
```
n01440764
n01695060
n01843383
...
```

## Usage
Once the environment is installed use the following command to run the test:
```
NUM_VAL_THREADS=8 pytest --data=<path_to_imagenet_val_folder> --output=./tmp tests/post_training/test_quantize_conformance.py
```

`NUM_VAL_THREADS` environment variable controls the number of parallel streams when validating the model.


