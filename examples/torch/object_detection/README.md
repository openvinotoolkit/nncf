# Object Detection sample

This sample demonstrates DL model compression capabilities for object detection task.

## Features:

- Vanilla SSD300 / SSD512 (+ Batch Normalization), MobileNetSSD-300
- VOC2007 / VOC2012, COCO datasets
- Configuration file examples for sparsity, quantization, filter pruning and quantization with sparsity
- Export to ONNX compatible with OpenVINO (compatible with pre-shipped CPU extensions detection layers)
- DataParallel and DistributedDataParallel modes
- Tensorboard output

## Installation

At this point it is assumed that you have already installed nncf. You can find information on downloading nncf [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the sample you should install the corresponding Python package dependencies:

```
pip install -r examples/torch/requirements.txt
```

## Quantize FP32 pretrained model

This scenario demonstrates quantization with fine-tuning of SSD300 on VOC dataset.

#### Dataset preparation

- Download and extract in one folder train/val+test VOC2007 and train/val VOC2012 data from [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
- In the future, `<path_to_dataset>` means the path to this folder.

#### Run object detection sample

- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable
- Navigate to the `examples/torch/object_detection` folder
- (Optional) Before compressing a model, it is highly recommended checking the accuracy of the pretrained model, use the following command:
  ```bash
  python main.py \
  --mode=test \
  --config=configs/ssd300_vgg_voc_int8.json \
  --data=<path_to_dataset> \
  --disable-compression
  ```
- Run the following command to start compression with fine-tuning on GPUs:
  `python main.py -m train --config configs/ssd300_vgg_voc_int8.json --data <path_to_dataset> --log-dir=../../results/quantization/ssd300_int8 --weights=<path_to_checkpoint>`It may take a few epochs to get the baseline accuracy results.
- Use `--weights` flag with the path to a compatible PyTorch checkpoint in order to load all matching weights from the checkpoint into the model - useful if you need to start compression-aware training from a previously trained uncompressed (FP32) checkpoint instead of performing compression-aware training from scratch. This flag is optional, but highly recommended to use.
- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a previously saved model to resume training.
- Use `--prepare-for-inference` argument to convert model to torch native format before `test` and `export` steps.

#### Validate your model checkpoint

To estimate the test scores of your trained model checkpoint use the following command:
`python main.py -m test --config=configs/ssd300_vgg_voc_int8.json --data <path_to_dataset> --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

#### Export compressed model

To export trained model to ONNX format use the following command:
`python main.py -m export --config configs/ssd300_vgg_voc_int8.json --data <path_to_dataset> --resume <path_to_compressed_model_checkpoint> --to-onnx=../../results/ssd300_int8.onnx`

#### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

<a name="results"></a>
### Results

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-MobileNet|None|VOC12+07 train, VOC07 eval|62.23|[ssd300_mobilenet_voc.json](configs/ssd300_mobilenet_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc)|
|SSD300-MobileNet|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|62.95 (-0.72)|[ssd300_mobilenet_voc_magnitude_int8.json](configs/ssd300_mobilenet_voc_magnitude_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8)|
|SSD300-VGG-BN|None|VOC12+07 train, VOC07 eval|78.28|[ssd300_vgg_voc.json](configs/ssd300_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc)|
|SSD300-VGG-BN|INT8|VOC12+07 train, VOC07 eval|77.81 (0.47)|[ssd300_vgg_voc_int8.json](configs/ssd300_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8)|
|SSD300-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|77.66 (0.62)|[ssd300_vgg_voc_magnitude_sparsity_int8.json](configs/ssd300_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8)|
|SSD512-VGG-BN|None|VOC12+07 train, VOC07 eval|80.26|[ssd512_vgg_voc.json](configs/ssd512_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc)|
|SSD512-VGG-BN|INT8|VOC12+07 train, VOC07 eval|80.04 (0.22)|[ssd512_vgg_voc_int8.json](configs/ssd512_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8)|
|SSD512-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|79.68 (0.58)|[ssd512_vgg_voc_magnitude_sparsity_int8.json](configs/ssd512_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8)|

<a name="filter_pruning"></a>
#### Results for filter pruning
|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-VGG-BN|None|VOC12+07 train, VOC07 eval|78.28|[ssd300_vgg_voc.json](configs/ssd300_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc)|
|SSD300-VGG-BN|Filter pruning, 40%, geometric median criterion|VOC12+07 train, VOC07 eval|78.35 (-0.07)|[ssd300_vgg_voc_pruning_geometric_median.json](configs/ssd300_vgg_voc_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median)|
|SSD300-VGG-BN|Filter pruning 40%,<br/>geometric median criterion|VOC12+07 train, VOC07 eval|77.72 (0.56)|25.8 (42.23%)|11.4 (43.35%)|[Link](configs/ssd300_vgg_voc_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth)|
