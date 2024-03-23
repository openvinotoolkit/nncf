# Semantic segmentation sample

This sample demonstrates DL model compression capabilities for semantic segmentation problem

## Features

- UNet and ICNet with implementations as close as possible to the original papers
- Loaders for CamVid, Cityscapes (20-class), Mapillary Vistas(20-class), Pascal VOC (reuses the loader integrated into torchvision)
- Configuration file examples for sparsity, quantization, filter pruning and quantization with sparsity
- Export to ONNX compatible with OpenVINO
- DataParallel and DistributedDataParallel modes
- Tensorboard output

## Installation

At this point it is assumed that you have already installed nncf. You can find information on downloading nncf [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the sample you should install the corresponding Python package dependencies:

```bash
pip install -r examples/torch/requirements.txt
```

## Quantize FP32 pretrained model

This scenario demonstrates quantization with fine-tuning of UNet on Mapillary Vistas dataset.

### Dataset preparation

- Obtain a copy of Mapillary Vistas train/val data [here](https://www.mapillary.com/dataset/vistas/)

### Run semantic segmentation sample

- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable
- Navigate to the `examples/torch/segmentation` folder
- (Optional) Before compressing a model, it is highly recommended checking the accuracy of the pretrained model, use the following command:

  ```bash
  python main.py \
  --mode=test \
  --config=configs/unet_mapillary_int8.json \
  --weights=<path_to_fp32_model_checkpoint> \
  --data=<path_to_dataset> \
  --batch-size=1 \
  --disable-compression
  ```

- Run the following command to start compression with fine-tuning on GPUs:
  `python main.py -m train --config configs/unet_mapillary_int8.json --data <path_to_dataset> --weights <path_to_fp32_model_checkpoint>`

It may take a few epochs to get the baseline accuracy results.

- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a model from the previous experiment to resume training.
- Use `-b <number>` option to specify the total batch size across GPUs
- Use the `--weights` flag with the path to a compatible PyTorch checkpoint in order to load all matching weights from the checkpoint into the model - useful
  if you need to start compression-aware training from a previously trained uncompressed (FP32) checkpoint instead of performing compression-aware training fr
  om scratch.
- Use `--export-model-path` to specify the path to export the model in OpenVINO or ONNX format by using the .xml or .onnx suffix, respectively.
- Use the `--no-strip-on-export` to export not stripped model.
- Use the `--export-to-ir-via-onnx` to to export to OpenVINO, will produce the serialized OV IR object by first exporting the torch model object to an .onnx file and then converting that .onnx file to an OV IR file.

### Validate your model checkpoint

To estimate the test scores of your trained model checkpoint use the following command:
`python main.py -m test --config=configs/unet_mapillary_int8.json --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

### Export compressed model

To export trained model to ONNX format use the following command:
`python main.py --mode export --config configs/unet_mapillary_int8.json --data <path_to_dataset> --resume <path_to_compressed_model_checkpoint> --to-ir ../../results`

### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

## Results

Please see compression results for PyTorch semantic segmentation at our [Model Zoo page](../../../docs/ModelZoo.md#pytorch-semantic-segmentation).
