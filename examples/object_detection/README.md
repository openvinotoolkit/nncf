# Object Detection sample
This sample demonstrates DL model compression capabailites for object detection task.

## Features:
- Vanilla SSD300 / SSD512 (+ Batch Normalization), MobileNetSSD-300
- VOC2007 / VOC2012, COCO datasets
- Configuration file examples for sparsity and quantization
- Export to ONNX compatible with OpenVINO (compatible with pre-shipped CPU extensions detection layers)
- DataParallel and DistributedDataParallel modes
- Tensorboard output

## Quantize FP32 pretrained model
This scenario demonstrates quantization with fine-tuning of SSD300 on VOC dataset.

#### Dataset preparation
- Download and extract VOC2007 and VOC2012 train/val and test data + devkit from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) and [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#devkit)

#### Run object detection sample
- If you did not install the package then add the repository root folder to the `PYTHONPATH` environment variable
- Navigate to the `examples/object_detection` folder
- Run the following command to start compression with fine-tuning on GPUs:
`python main.py -m train --config configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --log-dir=../../results/quantization/ssd300_int8`
It may take a few epochs to get the baseline accuracy results.
- Use `--multiprocessing-distributed` flag to run in the distributed mode.
- Use `--resume` flag with the path to a previously saved model to resume training.

#### Validate your model checkpoint
To estimate the test scores of your model checkpoint use the following command:
`python main.py -m test --config=configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --resume <path_to_trained_model_checkpoint>`
If you want to validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

#### Export compressed model
To export trained model to ONNX format use the following command:
`python main.py -m test --config configs/ssd300_vgg_int8_voc.json --data <path_to_dataset> --resume <path_to_compressed_model_checkpoint> --to-onnx=../../results/ssd300_int8.onnx`

#### Export to OpenVINO Intermediate Representation (IR)

To export a model to OpenVINO IR and run it using Intel Deep Learning Deployment Toolkit please refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results

|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|NNCF config file|PyTorch checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-MobileNet|None|VOC07+12 train, VOC12 eval|86.02|[ssd300_mobilenet_voc.json](configs/ssd300_mobilenet_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_mobilenet_voc.pth)|
|SSD300-MobileNet|INT8 + Sparsity 70% (Magnitude)|VOC07+12 train, VOC12 eval|85.82|[ssd300_mobilenet_voc_magnitude_int8.json](configs/ssd300_mobilenet_voc_magnitude_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth)|
|SSD300-BN|None|VOC07+12 train, VOC12 eval|96.53|[ssd300_vgg_voc.json](configs/ssd300_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_vgg_voc.pth)|
|SSD300-BN|INT8|VOC07+12 train, VOC12 eval|96.5|[ssd300_vgg_voc_int8.json](configs/ssd300_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_vgg_voc_int8.pth)|
|SSD300-BN|INT8 + Sparsity 70% (Magnitude)|VOC07+12 train, VOC12 eval|95.4|[ssd300_vgg_voc_magnitude_sparsity_int8.json](configs/ssd300_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_vgg_voc_magnitude_sparsity_int8.pth)|
|SSD300-BN|Filter pruning, 40%, geometric median criterion|VOC07+12 train, VOC12 eval|91.98|[ssd300_vgg_voc_pruning_geometric_median.json](configs/ssd300_vgg_voc_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd300_vgg_voc_pruning_geometric_median.pth)|
|SSD512-BN|None|VOC07+12 train, VOC12 eval|97.59|[ssd512_vgg_voc.json](configs/ssd512_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd512_vgg_voc.pth)|
|SSD512-BN|INT8|VOC07+12 train, VOC12 eval|97.37|[ssd512_vgg_voc_int8.json](configs/ssd512_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd512_vgg_voc_int8.pth)|
|SSD512-BN|INT8 + Sparsity 70% (Magnitude)|VOC07+12 train, VOC12 eval|96.66|[ssd512_vgg_voc_magnitude_sparsity_int8.json](configs/ssd512_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/v1.6.0/ssd512_vgg_voc_magnitude_sparsity_int8.pth)|
