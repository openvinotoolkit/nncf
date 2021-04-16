# Object Detection Sample

This sample demonstrates DL model compression capabilities for Object Detection task.
The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs and validation steps.
The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- RetinaNet from the official [TF repository](https://github.com/tensorflow/models/tree/master/official/vision/detection) with minor modifications (custom implementation of upsamling is replaced with equivalent tf.keras.layers.UpSampling2D). YOLOv4 from the [keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set) repository.
- Support [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) and TFRecords for COCO2017 dataset.
- Configuration file examples for sparsity, quantization, and quantization with sparsity.
- Export to Frozen Graph or TensorFlow SavedModel that is supported by the OpenVINO™ toolkit.
- Distributed training on multiple GPUs on one machine is supported using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).

## Quantize Pretrained Model

This scenario demonstrates quantization with fine-tuning of RetinaNet with ResNet-50 backbone on the COCO2017 dataset.

### Dataset Preparation

The object detection sample supports [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) and [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).
The dataset type is specified in the configuration file by setting the `"dataset_type"` parameter to `"tfds"` or `"tfrecords"` accordingly.
The TFDS format is used by default in the configuration file.

#### Using TFDS

Please read the following [guide](https://www.tensorflow.org/datasets/overview) for more information on how to use TFDS to download and prepare a dataset.
For the [COCO2017](https://www.tensorflow.org/datasets/catalog/coco#coco2017) dataset, TFDS supports automatic download.
All you need to do is to specify the dataset and its type in the configuration file as follows:

```json
    "dataset": "coco/2017",
    "dataset_type": "tfds"
```

#### Legacy TFRecords

To download the [COCO2017](https://cocodataset.org/) dataset and convert it to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
format please use [download_and_preprocess_coco.sh](https://github.com/tensorflow/tpu/blob/master/tools/datasets/download_and_preprocess_coco.sh)
script from the official [TensorFlow TPU](https://github.com/tensorflow/tpu) repository.

```bash
bash <path_to_tensorflow_tpu_repo>/tools/datasets/download_and_preprocess_coco.sh <path_to_coco_data_dir>
```

This script installs the required libraries and then runs the dataset preprocessing. The output of the script is `*.tfrecord` files in your local data directory.

The [COCO2017](https://cocodataset.org/) dataset in TFRecords format should be specified in the configuration file as follows:

```json
    "dataset": "coco/2017",
    "dataset_type": "tfrecords"
```

### Run Object Detection Sample

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/object_detection` folder.
- Download the pre-trained weights in H5 format and provide the path to them using `--weights` flag. The link to the 
archive with pre-trained weights can be found in the `TensorFlow checkpoint` column of the [results](#results) table. 
Select the checkpoint corresponding to the `None` compression algorithm, which includes the pre-trained weights for the 
FP32 model, without applying any compression algorithms.  
- Run the following command to start compression with fine-tuning on all available GPUs on the machine:
    ```bash
    python main.py \
    --mode=train \
    --config=configs/quantization/retinanet_coco_int8.json \
    --weights=<path_to_H5_file_with_pretrained_weights>
    --data=<path_to_dataset> \
    --log-dir=../../results/quantization/retinanet_coco_int8
    ```
- Use the `--resume` flag with the path to the checkpoint to resume training from the defined checkpoint or folder with checkpoints to resume training from the last checkpoint.

### Validate Your Model Checkpoint

To estimate the test scores of your model checkpoint, use the following command:
```bash
python main.py \
--mode=test \
--config=configs/quantization/retinanet_coco_int8.json \
--data=<path_to_dataset> \
--resume=<path_to_trained_model_checkpoint>
```

To validate an model checkpoint, make sure the compression algorithm settings are empty in the configuration file and path to`.h5` file with model weights is provided in command line argument `--weights`.

### Export Compressed Model

To export trained model to the **Frozen Graph**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/retinanet_coco_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-frozen-graph=../../results/retinanet_coco_int8.pb
```

To export trained model to the **SavedModel**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/retinanet_coco_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-saved-model=../../results/saved_model
```

To export trained model to the **Keras H5**, use the following command:
```bash
python main.py \
--mode=export \
--config=configs/quantization/retinanet_coco_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--to-h5=../../results/retinanet_coco_int8.h5
```

### Save Checkpoint without Optimizer
To reduce memory footprint (if no further training is scheduled) it is useful to save the checkpoint without optimizer. Use the following command:
```bash
python ../common/prepare_checkpoint.py \
--config=configs/quantization/retinanet_coco_int8.json \
--resume=<path_to_trained_model_checkpoint> \
--checkpoint-save-dir=<path_to_save_optimized_model_checkpoint>
```

### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

## Train RetinaNet from scratch
- Download pre-trained ResNet-50 checkpoint from [here](https://storage.cloud.google.com/cloud-tpu-checkpoints/model-garden-vision/detection/resnet50-2018-02-07.tar.gz).
- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/object_detection` folder.
- Run the following command to start training RetinaNet from scratch on all available GPUs on the machine:
    ```bash
    python main.py \
    --mode=train \
    --config=configs/retinanet_coco.json \
    --data=<path_to_dataset> \
    --log-dir=../../results/quantization/retinanet_coco_baseline \
    --backbone-checkpoint=<path_to_resnet50-2018-02-07_folder>
- Export trained model to the Keras H5 format.

<a name='results'></a>

## Results

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow mAP, %**|**NNCF config file**|**TensorFlow checkpoint**|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RetinaNet|None|COCO2017|33.44|[retinanet_coco.json](configs/retinanet_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/retinanet.tar.gz)|
|RetinaNet|INT8 w:sym,per-tensor a:sym,per-tensor |COCO2017|33.3|[retinanet_coco_int8.json](configs/quantization/retinanet_coco_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/retinanet_int8_w_sym_t_a_sym_t.tar.gz)|
|RetinaNet|Sparsity 50% (Magnitude)|COCO2017|33.13|[retinanet_coco_magnitude_sparsity.json](configs/sparsity/retinanet_coco_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/retinanet_sparsity_50.tar.gz)|
|RetinaNet|Filter Pruning 40%|COCO2017|32.7|[retinanet_coco_pruning.json](configs/pruning/retinanet_coco_pruning.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/retinanet_pruning_40.tar.gz)|
|YOLOv4|None|COCO2017|47.04|[yolo_v4_coco.json](configs/yolo_v4_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/yolo_v4.tar.gz)|
|YOLOv4|INT8 w:sym,per-channel a:asym,per-tensor |COCO2017|46.20|[yolo_v4_coco_int8.json](configs/quantization/yolo_v4_coco_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/yolo_v4_int8_w_sym_ch_a_asym_t.tar.gz)|
|YOLOv4|Sparsity 50% (Magnitude)|COCO2017|46.54|[yolo_v4_coco_magnitude_sparsity.json](configs/sparsity/yolo_v4_coco_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/tensorflow/models/develop/yolo_v4_sparsity_50.tar.gz)|
