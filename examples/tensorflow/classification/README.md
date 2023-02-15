# Image Classification Sample

This sample demonstrates a DL model compression in case of the Image Classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Models form the [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) module (ResNets, MobileNets, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support.
- Configuration file examples for sparsity, quantization, filter pruning and quantization with sparsity.
- Export to Frozen Graph or TensorFlow SavedModel that is supported by the OpenVINO™ toolkit.
- Distributed training on multiple GPUs on one machine is supported using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).

## Installation

At this point it is assumed that you have already installed nncf. You can find information on downloading nncf [here](https://github.com/openvinotoolkit/nncf#user-content-installation).  

To work with the sample you should install the corresponding Python package dependencies:

```
pip install -r examples/tensorflow/requirements.txt
```

## Quantize Pretrained Model

This scenario demonstrates quantization with fine-tuning of MobileNetV2 on the ImageNet dataset.

### Dataset Preparation

The classification sample supports [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) and [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).
The dataset type is specified in the configuration file by setting the `"dataset_type"` parameter to `"tfds"` or `"tfrecords"` accordingly. TFDS is used by default in all provided configuration files.

#### Using TFDS

Please read the following [guide](https://www.tensorflow.org/datasets/overview) for more information on how to use TFDS to download and prepare a dataset.

For the [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) dataset, TFDS requires a manual download. Please refer to the [TFDS ImageNet Readme](https://www.tensorflow.org/datasets/catalog/imagenet2012) for download instructions.
The TFDS ImageNet dataset should be specified in the configuration file as follows:
```json
    "dataset": "imagenet2012",
    "dataset_type": "tfds"
```

#### Legacy TFRecords

To download the [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) dataset and convert it to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format, refer to the following [tutorial](https://github.com/tensorflow/models/tree/master/research/slim#Data).
The ImageNet dataset in TFRecords format should be specified in the configuration file as follows:
```json
    "dataset": "imagenet2012",
    "dataset_type": "tfrecords"
```

### Run Classification Sample

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/classification` folder.

#### Test Pretrained Model

Before compressing a model, it is highly recommended checking the accuracy of the pretrained model. All models which are supported in the sample has pretrained weights for ImageNet. 

To load pretrained weights into a model and then evaluate the accuracy of that model, make sure that the pretrained=True option is set in the configuration file and use the following command:
```bash
python main.py \
--mode=test \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--data=<path_to_imagenet_dataset> \
--disable-compression 
```

#### Compress Pretrained Model

Run the following command to start compression with fine-tuning on all available GPUs on the machine:
  ```bash
  python main.py \
  --mode=train \
  --config=configs/quantization/mobilenet_v2_imagenet_int8.json \
  --data=<path_to_imagenet_dataset> \
  --log-dir=../../results/quantization/mobilenet_v2_int8
  ```
It may take a few epochs to get the baseline accuracy results.

Use the `--resume` flag with the path to the checkpoint to resume training from the defined checkpoint or folder with checkpoints to resume training from the last checkpoint.

### Validate Your Model Checkpoint

To estimate the test scores of your trained model checkpoint, use the following command:
```bash
python main.py \
--mode=test \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--data=<path_to_imagenet_dataset> \
--resume=<path_to_trained_model_checkpoint>
```

### Export Compressed Model

To export trained model to the **Frozen Graph**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-frozen-graph=../../results/mobilenet_v2_int8.pb
```

To export trained model to the **SavedModel**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-saved-model=../../results/saved_model
```

To export trained model to the **Keras H5**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-h5=../../results/mobilenet_v2_int8.h5
```

### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results
<a name="results"></a>

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Inception V3|None|ImageNet|77.91|[inception_v3_imagenet.json](configs/inception_v3_imagenet.json)|-|
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|78.39 (-0.48)|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_int8.tar.gz)|
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 61% (RB)|ImageNet|77.52 (0.39)|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_rb_sparsity_int8.tar.gz)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.86 (0.05)|[inception_v3_imagenet_magnitude_sparsity.json](configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_magnitude_sparsity.tar.gz)|
|MobileNet V2|None|ImageNet|71.85|[mobilenet_v2_imagenet.json](configs/mobilenet_v2_imagenet.json)|-|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|71.63 (0.22)|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_int8.tar.gz)|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 52% (RB)|ImageNet|70.94 (0.91)|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity_int8.tar.gz)|
|MobileNet V2| Sparsity 50% (RB)|ImageNet|71.34 (0.51)|[mobilenet_v2_imagenet_rb_sparsity.json](configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity.tar.gz)|
|MobileNet V2 (TensorFlow Hub MobileNet V2)|Sparsity 35% (Magnitude)|ImageNet|71.87 (-0.02)|[mobilenet_v2_hub_imagenet_magnitude_sparsity.json](configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_hub_imagenet_magnitude_sparsity.tar.gz)|
|MobileNet V3 (Small)|None|ImageNet|68.38|[mobilenet_v3_small_imagenet.json](configs/mobilenet_v3_small_imagenet.json)|-|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|67.79 (0.59)|[mobilenet_v3_small_imagenet_int8.json](configs/quantization/mobilenet_v3_small_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_int8.tar.gz)|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (Magnitude)|ImageNet|67.44 (0.94)|[mobilenet_v3_small_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_rb_sparsity_int8.tar.gz)|
|MobileNet V3 (Large)|None|ImageNet|75.80|[mobilenet_v3_large_imagenet.json](configs/mobilenet_v3_large_imagenet.json)|-|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|75.04 (0.76)|[mobilenet_v3_large_imagenet_int8.json](configs/quantization/mobilenet_v3_large_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_int8.tar.gz)|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (RB)|ImageNet|75.24 (0.56)|[mobilenet_v3_large_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_rb_sparsity_int8.tar.gz)|
|ResNet-50|None|ImageNet|75.05|[resnet50_imagenet.json](configs/resnet50_imagenet.json)|-|
|ResNet-50|INT8|ImageNet|74.99 (0.06)|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_int8.tar.gz)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 65% (RB)|ImageNet|74.36 (0.69)|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity_int8.tar.gz)|
|ResNet-50|Sparsity 80% (RB)|ImageNet|74.38 (0.67)|[resnet50_imagenet_rb_sparsity.json](configs/sparsity/resnet50_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity.tar.gz)|

#### Results for filter pruning
<a name="filter_pruning"></a>

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|75.05|[resnet50_imagenet.json](configs/resnet50_imagenet.json)|-|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|74.96 (0.09)|[resnet50_imagenet_pruning_geometric_median.json](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion|ImageNet|75.09 (-0.04)|[resnet50_imagenet_pruning_geometric_median_int8.json](configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz)|

#### Results for accuracy-aware compressed training
<a name="accuracy_aware"></a>

|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**NNCF config file**|
| :---: | :---: | :---: | :---: | :---: |
|ResNet50|Sparsity 65% (magnitude)|ImageNet|74.37 (0.67)|[resnet50_imagenet_magnitude_sparsity_accuracy_aware.json](configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json)|
