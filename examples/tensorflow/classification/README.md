# Image Classification Sample

This sample demonstrates a DL model compression in case of the Image Classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Models form the [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) module (ResNets, MobileNets, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support.
- Configuration file examples for sparsity, quantization, filter pruning and quantization with sparsity.
- Export to Frozen Graph or TensorFlow SavedModel that is supported by the OpenVINO™ toolkit.
- Distributed training on multiple GPUs on one machine is supported using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).

## Installation

To work with the sample you should install the corresponding Python package dependencies

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
- Run the following command to start compression with fine-tuning on all available GPUs on the machine:
    ```bash
    python main.py \
    --mode=train \
    --config=configs/quantization/mobilenet_v2_imagenet_int8.json \
    --data=<path_to_imagenet_dataset> \
    --log-dir=../../results/quantization/mobilenet_v2_int8
    ```
    It may take a few epochs to get the baseline accuracy results.
- Use the `--resume` flag with the path to the checkpoint to resume training from the defined checkpoint or folder with checkpoints to resume training from the last checkpoint.

### Validate Your Model Checkpoint

To estimate the test scores of your model checkpoint, use the following command:
```bash
python main.py \
--mode=test \
--config=configs/quantization/mobilenet_v2_imagenet_int8.json \
--data=<path_to_imagenet_dataset> \
--resume=<path_to_trained_model_checkpoint>
```
To validate an model checkpoint, make sure the compression algorithm settings are empty in the configuration file and `pretrained=True` is set.

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

|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**NNCF config file**|**TensorFlow checkpoint**|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Inception V3|INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations) |ImageNet|78.35 (-0.45)|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/inception_v3_int8_w_sym_t_half_a_sym_t.tar.gz)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.87 (0.03)|[inception_v3_imagenet_magnitude_sparsity.json](configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/inception_v3_sparsity_54.tar.gz)|
|Inception V3|INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations) + Sparsity 61% (RB)|ImageNet|77.58 (0.32)|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/inception_v3_int8_w_sym_t_half_a_sym_t_rb_sparsity_61.tar.gz)|
|MobileNet V2|INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations) |ImageNet|71.66 (0.19)|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v2_int8_w_sym_t_half_a_sym_t.tar.gz)|
|MobileNet V2|Sparsity 50% (RB)|ImageNet|71.34 (0.51)|[mobilenet_v2_imagenet_rb_sparsity.json](configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v2_rb_sparsity_50.tar.gz)|
|MobileNet V2|int8(per-tensor, symmetric for weights; per-tensor, symmetric for activations) + sparsity 52% (RB)|ImageNet|71.0 (0.85)|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v2_int8_w_sym_t_half_a_sym_t_rb_sparsity_52.tar.gz)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|67.7 (0.68)|[mobilenet_v3_small_imagenet_int8.json](configs/quantization/mobilenet_v3_small_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v3_small_int8_w_sym_ch_half_a_asym_t.tar.gz)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|67.7 (0.68)|[mobilenet_v3_small_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v3_small_int8_w_sym_ch_half_a_asym_t_rb_sparsity_42.tar.gz)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|75.0 (0.81)|[mobilenet_v3_large_imagenet_int8.json](configs/quantization/mobilenet_v3_large_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v3_large_int8_w_sym_ch_half_a_asym_t.tar.gz)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|75.15 (0.66)|[mobilenet_v3_large_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/mobilenet_v3_large_int8_w_sym_ch_half_a_asym_t_sparsity_42.tar.gz)|
|ResNet50|INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations)|ImageNet|75.0 (0.04)|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/resnet50_int8_w_sym_t_half_a_sym_t.tar.gz)|
|ResNet50|Sparsity 80% (RB)|ImageNet|74.36 (0.68)|[resnet50_imagenet_rb_sparsity.json](configs/sparsity/resnet50_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/resnet50_rb_sparsity_80.tar.gz)|
|ResNet50|INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations) + Sparsity 65% (RB)|ImageNet|74.3 (0.74)|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/resnet50_int8_w_sym_t_half_a_sym_t_rb_sparsity_65.tar.gz)|
|TensorFlow Hub MobileNet V2|Sparsity 35% (Magnitude)|ImageNet|71.90 (-0.06)|[mobilenet_v2_hub_imagenet_magnitude_sparsity.json](configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/tf1_mobilenet_v2_1.0_224_sparsity_35.tar.gz)|

#### Results for filter pruning
|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**GFLOPS**|**MParams**|**NNCF config file**|**TensorFlow checkpoint**|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet50|None|ImageNet|75.04|7.75 (100%)|25.5 (100%)|-|-|
|ResNet50|Filter Pruning 40%, geometric_median criterion|ImageNet|74.98 (0.06)|4.29 (55.35%)|15.8 (61.96%)|[Link](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/resnet50_pruning_40.tar.gz)|
|ResNet50|Filter Pruning 40%, geometric_median criterion + INT8 (per-tensor, symmetric for weights; per-tensor, symmetric for activations)|ImageNet|75.08 (-0.04)|4.27 (55.10%)|15.8 (61.96%)|[Link](configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/v2.0.0/tensorflow/resnet50_int8_w_sym_t_half_a_sym_t_pruning_40.tar.gz)|

#### Results for accuracy-aware compressed training

|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**NNCF config file**|
| :---: | :---: | :---: | :---: | :---: |
|ResNet50|Sparsity 75% (magnitude)|ImageNet|74.42 (0.62)|[resnet50_imagenet_magnitude_sparsity_accuracy_aware.json](configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json)|
