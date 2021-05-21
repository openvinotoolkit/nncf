# Semantic segmentation sample
This sample demonstrates DL model compression capabilities for semantic segmentation problem

## Features:
- UNet and ICNet with implementations as close as possible to the original papers
- Loaders for CamVid, Cityscapes (20-class), Mapillary Vistas(20-class), Pascal VOC (reuses the loader integrated into torchvision)
- Configuration file examples for sparsity and quantization
- Export to ONNX compatible with OpenVINO
- DataParallel and DistributedDataParallel modes
- Tensorboard output

## Installation

To work with the sample you should install the corresponding Python package dependencies

```
pip install -r examples/torch/requirements.txt
```

## Quantize FP32 pretrained model
This scenario demonstrates quantization with fine-tuning of UNet on Mapillary Vistas dataset.

#### Dataset preparation
- Obtain a copy of Mapillary Vistas train/val data [here](https://www.mapillary.com/dataset/vistas/)

#### Run semantic segmentation sample
- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable
- Navigate to the `examples/torch/segmentation` folder
- Run the following command to start compression with fine-tuning on GPUs:
`python main.py -m train --config configs/unet_mapillary_int8.json --data <path_to_dataset> --weights <path_to_fp32_model_checkpoint>`

It may take a few epochs to get the baseline accuracy results.
- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a model from the previous experiment to resume training.
- Use `-b <number>` option to specify the total batch size across GPUs
- Use the `--weights` flag with the path to a compatible PyTorch checkpoint in order to load all matching weights from the checkpoint into the model - useful
 if you need to start compression-aware training from a previously trained uncompressed (FP32) checkpoint instead of performing compression-aware training fr
om scratch.


#### Validate your model checkpoint
To estimate the test scores of your model checkpoint use the following command:
`python main.py -m test --config=configs/unet_mapillary_int8.json --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

#### Export compressed model
To export trained model to ONNX format use the following command:
`python main.py --mode test --config configs/unet_mapillary_int8.json --data <path_to_dataset> --resume <path_to_compressed_model_checkpoint> --to-onnx unet_int8.onnx`

#### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results

|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|NNCF config file|PyTorch checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|UNet|None|CamVid|71.95|[unet_camvid.json](configs/unet_camvid.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_camvid.pth)|
|UNet|INT8|CamVid|71.8|[unet_camvid_int8.json](configs/unet_camvid_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_camvid_int8.pth)|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|72.03|[unet_camvid_magnitude_sparsity_int8.json](configs/unet_camvid_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_camvid_magnitude_sparsity_int8.pth)|
|ICNet|None|CamVid|67.89|[icnet_camvid.json](configs/icnet_camvid.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/icnet_camvid.pth)|
|ICNet|INT8|CamVid|67.86|[icnet_camvid_int8.json](configs/icnet_camvid_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/icnet_camvid_int8.pth)|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.18|[icnet_camvid_magnitude_sparsity_int8.json](configs/icnet_camvid_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/icnet_camvid_magnitude_sparsity_int8.pth)|
|UNet|None|Mapillary|56.23|[unet_mapillary.json](configs/unet_mapillary.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_mapillary.pth)|
|UNet|INT8|Mapillary|55.87|[unet_mapillary_int8.json](configs/unet_mapillary_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_mapillary_int8.pth)|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|55.65|[unet_mapillary_magnitude_sparsity_int8.json](configs/unet_mapillary_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_mapillary_magnitude_sparsity_int8.pth)|
|UNet|Filter pruning, 25%, geometric median criterion|Mapillary|55.62|[unet_mapillary_pruning_geometric_median.json](configs/unet_mapillary_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/unet_mapillary_pruning_geometric_median.pth)|
