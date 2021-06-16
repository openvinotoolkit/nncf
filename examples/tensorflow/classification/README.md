# Image Classification Sample

This sample demonstrates a DL model compression in case of the Image Classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Models form the [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) module (ResNets, MobileNets, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support.
- Configuration file examples for sparsity, quantization, and quantization with sparsity.
- Export to Frozen Graph or TensorFlow SavedModel that is supported by the OpenVINO™ toolkit.
- Distributed training on multiple GPUs on one machine is supported using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).

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

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow compressed accuracy**|**NNCF config file**|**TensorFlow checkpoint**|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Inception V3|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|78.35|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/inception_v3_int8_w_sym_t_half_a_sym_t.tar.gz)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.87|[inception_v3_imagenet_magnitude_sparsity.json](configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/inception_v3_sparsity_54.tar.gz)|
|Inception V3|INT8 w:sym,per-tensor a:sym,per-tensor + Sparsity 61% (RB)|ImageNet|77.58|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/inception_v3_int8_w_sym_t_half_a_sym_t_rb_sparsity_61.tar.gz)|
|MobileNet V2|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|71.66|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v2_int8_w_sym_t_half_a_sym_t.tar.gz)|
|MobileNet V2|Sparsity 50% (RB)|ImageNet|71.34|[mobilenet_v2_imagenet_rb_sparsity.json](configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v2_rb_sparsity_50.tar.gz)|
|MobileNet V2|int8 w:sym,per-tensor a:sym,per-tensor + sparsity 52% (RB)|ImageNet|71.0|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v2_int8_w_sym_t_half_a_sym_t_rb_sparsity_52.tar.gz)|
|MobileNet V3 small|INT8 w:sym,per-channel a:asym,per-tensor |ImageNet|67.7|[mobilenet_v3_small_imagenet_int8.json](configs/quantization/mobilenet_v3_small_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v3_small_int8_w_sym_ch_half_a_asym_t.tar.gz)|
|MobileNet V3 small|INT8 w:sym,per-channel a:asym,per-tensor + Sparsity 42% (RB)|ImageNet|67.7|[mobilenet_v3_small_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v3_small_int8_w_sym_ch_half_a_asym_t_rb_sparsity_42.tar.gz)|
|MobileNet V3 large|INT8 w:sym,per-channel a:asym,per-tensor |ImageNet|75.0|[mobilenet_v3_large_imagenet_int8.json](configs/quantization/mobilenet_v3_large_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v3_large_int8_w_sym_ch_half_a_asym_t.tar.gz)|
|MobileNet V3 large|INT8 w:sym,per-channel a:asym,per-tensor + Sparsity 42% (RB)|ImageNet|75.15|[mobilenet_v3_large_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/mobilenet_v3_large_int8_w_sym_ch_half_a_asym_t_sparsity_42.tar.gz)|
|ResNet50|INT8 w:sym,per-tensor a:sym,per-tensor|ImageNet|75.0|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/resnet50_int8_w_sym_t_half_a_sym_t.tar.gz)|
|ResNet50|Sparsity 80% (RB)|ImageNet|74.36|[resnet50_imagenet_rb_sparsity.json](configs/sparsity/resnet50_imagenet_rb_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/resnet50_rb_sparsity_80.tar.gz)|
|ResNet50|INT8 w:sym,per-tensor a:sym,per-tensor + Sparsity 65% (RB)|ImageNet|74.3|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/resnet50_int8_w_sym_t_half_a_sym_t_rb_sparsity_65.tar.gz)|
|ResNet50|Filter Pruning 40%|ImageNet|74.98|[resnet50_imagenet_pruning_geometric_median.json](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/resnet50_pruning_40.tar.gz)|
|TensorFlow Hub MobileNet V2|Sparsity 35% (Magnitude)|ImageNet|71.90|[mobilenet_v2_hub_imagenet_magnitude_sparsity.json](configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/tf1_mobilenet_v2_1.0_224_sparsity_35.tar.gz)|