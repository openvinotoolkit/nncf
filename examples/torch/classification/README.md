# Image Classification Sample

This sample demonstrates a DL model compression in case of an image-classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs, training and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Torchvision models (ResNets, VGG, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support
- Custom models support
- Configuration file examples for sparsity, quantization, filter pruning and quantization with sparsity
- Export to ONNX that is supported by the OpenVINO™ toolkit
- DataParallel and DistributedDataParallel modes
- Tensorboard-compatible output

## Installation

At this point it is assumed that you have already installed nncf. You can find information on downloading nncf [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the sample you should install the corresponding Python package dependencies:

```bash
pip install -r examples/torch/requirements.txt
```

## Quantize FP32 Pretrained Model

This scenario demonstrates quantization with fine-tuning of MobileNet v2 on the ImageNet dataset.

### Dataset Preparation

To prepare the ImageNet dataset, refer to the following [tutorial](https://github.com/pytorch/examples/tree/master/imagenet).

### Run Classification Sample

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/torch/classification` folder.

### Test Pretrained Model

Before compressing a model, it is highly recommended checking the accuracy of the pretrained model. All models which are supported in the sample has pretrained weights for ImageNet.

To load pretrained weights into a model and then evaluate the accuracy of that model, make sure that the pretrained=True option is set in the configuration file and use the following command:

```bash
python main.py \
--mode=test \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--data=<path_to_imagenet_dataset> \
--disable-compression
```

### Compress Pretrained Model

- Run the following command to start compression with fine-tuning on GPUs:

  ```bash
  python main.py -m train \
  --config configs/quantization/mobilenet_v2_imagenet_int8.json \
  --data /data/imagenet/ \
  --log-dir=../../results/quantization/mobilenet_v2_int8/
  ```

  It may take a few epochs to get the baseline accuracy results.
- Use the `--multiprocessing-distributed` flag to run in the distributed mode.
- Use the `--resume` flag with the path to a previously saved model to resume training.
- For Torchvision-supported image classification models, set `"pretrained": true` inside the NNCF config JSON file supplied via `--config` to initialize the model to be compressed with Torchvision-supplied pretrained weights, or, alternatively:
- Use the `--weights` flag with the path to a compatible PyTorch checkpoint in order to load all matching weights from the checkpoint into the model - useful if you need to start compression-aware training from a previously trained uncompressed (FP32) checkpoint instead of performing compression-aware training from scratch.
- Use `--export-model-path` to specify the path to export the model in OpenVINO or ONNX format by using the .xml or .onnx suffix, respectively.
- Use the `--no-strip-on-export` to export not stripped model.
- Use the `--export-to-ir-via-onnx` to to export to OpenVINO, will produce the serialized OV IR object by first exporting the torch model object to an .onnx file and then converting that .onnx file to an OV IR file.

### Validate Your Model Checkpoint

To estimate the test scores of your trained model checkpoint, use the following command:

```bash
python main.py -m test \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--resume <path_to_trained_model_checkpoint>
```

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

### Export Compressed Model

To export trained model to the ONNX format, use the following command:

```bash
python main.py -m export \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--resume=../../results/quantization/mobilenet_v2_int8/6/checkpoints/epoch_1.pth \
--to-ir=../../results
```

### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

## Results

Please see compression results for PyTorch classification at our [Model Zoo page](../../../docs/ModelZoo.md#pytorch-classification).
