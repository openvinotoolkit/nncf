# Instance Segmentation Sample

This sample demonstrates DL model compression capabilities for Instance Segmentation task.
The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs and validation steps.
The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Mask R-CNN from the official [TF repository](https://github.com/tensorflow/models/tree/master/official/legacy/detection) with minor modifications (custom implementation of upsampling is replaced with equivalent tf.keras.layers.UpSampling2D).
- Support TFRecords for COCO2017 dataset.
- Configuration file examples for sparsity, quantization, and quantization with sparsity.
- Export to Frozen Graph or TensorFlow SavedModel that is supported by the OpenVINO™ toolkit.
- Distributed training on multiple GPUs on one machine is supported using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).

## Installation

At this point it is assumed that you have already installed nncf. You can find information on downloading nncf [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the sample you should install the corresponding Python package dependencies:

```bash
pip install -r examples/tensorflow/requirements.txt
```

## Quantize Pretrained Model

This scenario demonstrates quantization with fine-tuning of Mask R-CNN with ResNet-50 backbone on the COCO2017 dataset.

### Dataset Preparation

The instance segmentation sample supports dataset only in [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) format.

To download the [COCO2017](https://cocodataset.org/) dataset and convert it to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
format please use [download_and_preprocess_coco.sh](https://github.com/tensorflow/tpu/blob/master/tools/datasets/download_and_preprocess_coco.sh)
script from the official [TensorFlow TPU](https://github.com/tensorflow/tpu) repository.

```bash
bash <path_to_tensorflow_tpu_repo>/tools/datasets/download_and_preprocess_coco.sh <path_to_coco_data_dir>
```

This script installs the required libraries and then runs the dataset preprocessing. The output of the script is `*.tfrecord` files in your local data directory.

The [COCO2017](https://cocodataset.org/) dataset should be specified in the configuration file as follows:

```json
"dataset": "coco/2017"
```

### Run Instance Segmentation Sample

We can run the sample after data preparation. For this follow these steps:

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/segmentation` folder.
- Download the pre-trained Mask-R-CNN [weights](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz) in checkpoint format and provide the path to them using `--weights` flag.
- Specify the GPUs to be used for training by setting the environment variable [`CUDA_VISIBLE_DEVICES`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/). This is necessary because training and validation during training must be performed on different GPU devices. Please note that usually only one GPU is required for validation during training.
- (Optional) Before compressing a model, it is highly recommended checking the accuracy of the pretrained model, use the following command:

  ```bash
  python evaluation.py \
  --mode=test \
  --config=configs/quantization/mask_rcnn_coco_int8.json \
  --weights=<path_to_ckpt_file_with_pretrained_weights> \
  --data=<path_to_dataset> \
  --batch-size=1 \
  --disable-compression
  ```

- Run the following command to start compression with fine-tuning on all available GPUs on the machine:

  ```bash
  python train.py \
  --config=configs/quantization/mask_rcnn_coco_int8.json \
  --weights=<path_to_ckpt_file_with_pretrained_weights> \
  --data=<path_to_dataset> \
  --log-dir=../../results/quantization/maskrcnn_coco_int8
  ```

- Use the `--resume` flag with the path to the checkpoint to resume training from the defined checkpoint or folder with checkpoints to resume training from the last checkpoint.

To start checkpoints validation during training follow these steps:

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/segmentation` folder.
- Specify the GPUs to be used for validation during training by setting the environment variable [`CUDA_VISIBLE_DEVICES`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/).
- Run the following command to start checkpoints validation during training:

  ```bash
  python evaluation.py \
  --mode=train \
  --config=configs/quantization/mask_rcnn_coco_int8.json \
  --data=<path_to_dataset> \
  --batch-size=1 \
  --checkpoint-save-dir=<path_to_checkpoints>
  ```

### Validate Your Model Checkpoint

To estimate the test scores of your trained model checkpoint, use the following command:

```bash
python evaluation.py \
--mode=test \
--config=configs/quantization/mask_rcnn_coco_int8.json \
--data=<path_to_dataset> \
--batch-size=1 \
--resume=<path_to_trained_model_checkpoint>
```

### Export Compressed Model

To export trained model to the **Frozen Graph**, use the following command:

```bash
python evaluation.py \
--mode=export \
--config=configs/quantization/mask_rcnn_coco_int8.json \
--batch-size=1 \
--resume=<path_to_trained_model_checkpoint> \
--to-frozen-graph=../../results/mask_rcnn_coco_int8.pb
```

To export trained model to the **SavedModel**, use the following command:

```bash
python evaluation.py \
--mode=export \
--config=configs/quantization/mask_rcnn_coco_int8.json \
--batch-size=1 \
--resume=<path_to_trained_model_checkpoint> \
--to-saved-model=../../results/saved_model
```

### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

## Train MaskRCNN from scratch

- Download pre-trained ResNet-50 checkpoint from [here](https://storage.cloud.google.com/cloud-tpu-checkpoints/model-garden-vision/detection/resnet50-2018-02-07.tar.gz).
- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/tensorflow/segmentation` folder.
- Run the following command to start training MaskRCNN from scratch on all available GPUs on the machine:

  ```bash
  python train.py \
  --config=configs/mask_rcnn_coco.json \
  --backbone-checkpoint=<path_to_resnet50-2018-02-07_folder> \
  --data=<path_to_dataset> \
  --log-dir=../../results/quantization/maskrcnn_coco_baseline
  ```

## Results

Please see compression results for Tensorflow instance segmentation at our [Model Zoo page](../../../docs/ModelZoo.md#tensorflow-instance-segmentation).
