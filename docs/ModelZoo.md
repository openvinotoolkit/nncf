# NNCF Compressed Model Zoo

Results achieved using sample scripts, example patches to third-party repositories and NNCF configuration files provided
with this repository. See README.md files for [sample scripts](#model-compression-samples) and [example patches](#third-party-repository-integration)
to find instruction and links to exact configuration files and final checkpoints.
- [PyTorch models](#pytorch-models)
  * [Classification](#pytorch-classification)
  * [Object detection](#pytorch-object-detection)
  * [Semantic segmentation](#pytorch_semantic_segmentation)
  * [Natural language processing (3rd-party training pipelines)](#pytorch_nlp)
- [TensorFlow models](#tensorflow-models)
  * [Classification](#tensorflow_classification)
  * [Object detection](#tensorflow_object_detection)
  * [Instance segmentation](#tensorflow_instance_segmentation)
- [ONNX models](#onnx-models)

<a name="pytorch_models"></a>
## PyTorch

<a name="pytorch_classification"></a>
### Classification

#### Quantization

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.15|[resnet50_imagenet.json](configs/quantization/resnet50_imagenet.json)|-|
|ResNet-50|INT8|ImageNet|76.46 (-0.31)|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8.pth)|
|ResNet-50|INT8 (per-tensor only)|ImageNet|76.39 (-0.24)|[resnet50_imagenet_int8_per_tensor.json](configs/quantization/resnet50_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8_per_tensor.pth)|
|ResNet-50|Mixed, 43.12% INT8 / 56.88% INT4|ImageNet|76.05 (0.10)|[resnet50_imagenet_mixed_int_hawq.json](configs/mixed_precision/resnet50_imagenet_mixed_int_hawq.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int4_int8.pth)|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|75.42 (0.73)|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity_int8.pth)|
|ResNet-50|INT8 + Sparsity 50% (RB)|ImageNet|75.50 (0.65)|[resnet50_imagenet_rb_sparsity50_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity50_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity50_int8.pth)|
|Inception V3|None|ImageNet|77.33|[inception_v3_imagenet.json](configs/quantization/inception_v3_imagenet.json)|-|
|Inception V3|INT8|ImageNet|77.45 (-0.12)|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_int8.pth)|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|76.36 (0.97)|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_rb_sparsity_int8.pth)|
|MobileNet V2|None|ImageNet|71.87|[mobilenet_v2_imagenet.json](configs/quantization/mobilenet_v2_imagenet.json)|-|
|MobileNet V2|INT8|ImageNet|71.07 (0.80)|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8.pth)|
|MobileNet V2|INT8 (per-tensor only)|ImageNet|71.24 (0.63)|[mobilenet_v2_imagenet_int8_per_tensor.json](configs/quantization/mobilenet_v2_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8_per_tensor.pth)|
|MobileNet V2|Mixed, 58.88% INT8 / 41.12% INT4|ImageNet|70.95 (0.92)|[mobilenet_v2_imagenet_mixed_int_hawq.json](configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_hawq.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int4_int8.pth)|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.09 (0.78)|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_rb_sparsity_int8.pth)|
|MobileNet V3 small|None|ImageNet|67.66|[mobilenet_v3_small_imagenet.json](configs/quantization/mobilenet_v3_small_imagenet.json)|-|
|MobileNet V3 small|INT8|ImageNet|66.98 (0.68)|[mobilenet_v3_small_imagenet_int8.json](configs/quantization/mobilenet_v3_small_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v3_small_imagenet_int8.pth)|
|SqueezeNet V1.1|None|ImageNet|58.19|[squeezenet1_1_imagenet.json](configs/quantization/squeezenet1_1_imagenet.json)|-|
|SqueezeNet V1.1|INT8|ImageNet|58.22 (-0.03)|[squeezenet1_1_imagenet_int8.json](configs/quantization/squeezenet1_1_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8.pth)|
|SqueezeNet V1.1|INT8 (per-tensor only)|ImageNet|58.11 (0.08)|[squeezenet1_1_imagenet_int8_per_tensor.json](configs/quantization/squeezenet1_1_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8_per_tensor.pth)|
|SqueezeNet V1.1|Mixed, 52.83% INT8 / 47.17% INT4|ImageNet|57.57 (0.62)|[squeezenet1_1_imagenet_mixed_int_hawq_old_eval.json](configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq_old_eval.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int4_int8.pth)|
|ResNet-18|None|ImageNet|69.76|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|-|
|ResNet-34|None|ImageNet|73.30|[resnet34_imagenet.json](configs/pruning/resnet34_imagenet.json)|-|
|GoogLeNet|None|ImageNet|69.77|[googlenet_imagenet.json](configs/pruning/googlenet_imagenet.json)|-|

#### Binarization

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-18|None|ImageNet|69.76|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|-|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.67 (8.09)|[resnet18_imagenet_binarization_xnor.json](configs/binarization/resnet18_imagenet_binarization_xnor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_xnor.pth)|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.63 (8.13)|[resnet18_imagenet_binarization_dorefa.json](configs/binarization/resnet18_imagenet_binarization_dorefa.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_dorefa.pth)|

#### Filter pruning

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.15|[resnet50_imagenet.json](configs/quantization/resnet50_imagenet.json)|-|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|75.57 (0.58)|[resnet50_imagenet_pruning_geometric_median.json](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_pruning_geometric_median.pth)|
|ResNet-18|None|ImageNet|69.76|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|-|
|ResNet-18|Filter pruning, 40%, magnitude criterion|ImageNet|69.27 (0.49)|[resnet18_imagenet_pruning_magnitude.json](configs/pruning/resnet18_imagenet_pruning_magnitude.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_magnitude.pth)|
|ResNet-18|Filter pruning, 40%, geometric median criterion|ImageNet|69.31 (0.45)|[resnet18_imagenet_pruning_geometric_median.json](configs/pruning/resnet18_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_geometric_median.pth)|
|ResNet-34|None|ImageNet|73.30|[resnet34_imagenet.json](configs/pruning/resnet34_imagenet.json)|-|
|ResNet-34|Filter pruning, 50%, geometric median criterion + KD|ImageNet|73.11 (0.19)|[resnet34_imagenet_pruning_geometric_median_kd.json](configs/pruning/resnet34_imagenet_pruning_geometric_median_kd.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet34_imagenet_pruning_geometric_median_kd.pth)|
|GoogLeNet|None|ImageNet|69.77|[googlenet_imagenet.json](configs/pruning/googlenet_imagenet.json)|-|
|GoogLeNet|Filter pruning, 40%, geometric median criterion|ImageNet|69.47 (0.30)|[googlenet_imagenet_pruning_geometric_median.json](configs/pruning/googlenet_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/googlenet_imagenet_pruning_geometric_median.pth)|

#### Accuracy-aware compressed training
|Model|Compression algorithm|Dataset|Accuracy (Drop) %|NNCF config file|
| :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.16|[resnet50_imagenet.json](configs/quantization/resnet50_imagenet.json)|
|ResNet-50|Filter pruning, 52.5%, geometric median criterion|ImageNet|75.23 (0.93)|[resnet50_imagenet_accuracy_aware.json](configs/pruning/resnet50_imagenet_pruning_accuracy_aware.json)|
|ResNet-18|None|ImageNet|69.8|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|
|ResNet-18|Filter pruning, 60%, geometric median criterion|ImageNet|69.2 (-0.6)|[resnet18_imagenet_accuracy_aware.json](configs/pruning/resnet18_imagenet_pruning_accuracy_aware.json)|


<a name="pytorch_object_detection"></a>
### Object detection

#### Quantization

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-MobileNet|None|VOC12+07 train, VOC07 eval|62.23|[ssd300_mobilenet_voc.json](configs/ssd300_mobilenet_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc.pth)|
|SSD300-MobileNet|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|62.95 (-0.72)|[ssd300_mobilenet_voc_magnitude_int8.json](configs/ssd300_mobilenet_voc_magnitude_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth)|
|SSD300-VGG-BN|None|VOC12+07 train, VOC07 eval|78.28|[ssd300_vgg_voc.json](configs/ssd300_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth)|
|SSD300-VGG-BN|INT8|VOC12+07 train, VOC07 eval|77.81 (0.47)|[ssd300_vgg_voc_int8.json](configs/ssd300_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8.pth)|
|SSD300-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|77.66 (0.62)|[ssd300_vgg_voc_magnitude_sparsity_int8.json](configs/ssd300_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8.pth)|
|SSD512-VGG-BN|None|VOC12+07 train, VOC07 eval|80.26|[ssd512_vgg_voc.json](configs/ssd512_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc.pth)|
|SSD512-VGG-BN|INT8|VOC12+07 train, VOC07 eval|80.04 (0.22)|[ssd512_vgg_voc_int8.json](configs/ssd512_vgg_voc_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8.pth)|
|SSD512-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|79.68 (0.58)|[ssd512_vgg_voc_magnitude_sparsity_int8.json](configs/ssd512_vgg_voc_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8.pth)|


#### Filter pruning
|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|SSD300-VGG-BN|None|VOC12+07 train, VOC07 eval|78.28|[ssd300_vgg_voc.json](configs/ssd300_vgg_voc.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth)|
|SSD300-VGG-BN|Filter pruning, 40%, geometric median criterion|VOC12+07 train, VOC07 eval|78.35 (-0.07)|[ssd300_vgg_voc_pruning_geometric_median.json](configs/ssd300_vgg_voc_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth)|


<a name="pytorch_semantic_segmentation"></a>
### Semantic segmentation

#### Quantization

|Model|Compression algorithm|Dataset|mIoU (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|UNet|None|CamVid|71.95|[unet_camvid.json](configs/unet_camvid.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid.pth)|
|UNet|INT8|CamVid|71.89 (0.06)|[unet_camvid_int8.json](configs/unet_camvid_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_int8.pth)|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|72.46 (-0.51)|[unet_camvid_magnitude_sparsity_int8.json](configs/unet_camvid_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_magnitude_sparsity_int8.pth)|
|ICNet|None|CamVid|67.89|[icnet_camvid.json](configs/icnet_camvid.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid.pth)|
|ICNet|INT8|CamVid|67.89 (0.00)|[icnet_camvid_int8.json](configs/icnet_camvid_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_int8.pth)|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.16 (0.73)|[icnet_camvid_magnitude_sparsity_int8.json](configs/icnet_camvid_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_magnitude_sparsity_int8.pth)|
|UNet|None|Mapillary|56.24|[unet_mapillary.json](configs/unet_mapillary.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth)|
|UNet|INT8|Mapillary|56.09 (0.15)|[unet_mapillary_int8.json](configs/unet_mapillary_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_int8.pth)|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|55.69 (0.55)|[unet_mapillary_magnitude_sparsity_int8.json](configs/unet_mapillary_magnitude_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_magnitude_sparsity_int8.pth)|


#### Filter pruning
|Model|Compression algorithm|Dataset|mIoU (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|UNet|None|Mapillary|56.24|[unet_mapillary.json](configs/unet_mapillary.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth)|
|UNet|Filter pruning, 25%, geometric median criterion|Mapillary|55.64 (0.60)|[unet_mapillary_pruning_geometric_median.json](configs/unet_mapillary_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_pruning_geometric_median.pth)|


<a name="pytorch_nlp"></a>
### NLP (HuggingFace Transformers-powered models)

|PyTorch Model|<img width="20" height="1">Compression algorithm<img width="20" height="1">|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|BERT-base-chinese|INT8|XNLI|77.22 (0.46)|
|BERT-base-cased|INT8|CoNLL2003|99.18 (-0.01)|
|BERT-base-cased|INT8|MRPC|84.8 (-0.24)|
|BERT-large (Whole Word Masking)|INT8|SQuAD v1.1|F1: 92.68 (0.53)|
|RoBERTa-large|INT8|MNLI|matched: 89.25 (1.35)|
|DistilBERT-base|INT8|SST-2|90.3 (0.8)|
|MobileBERT|INT8|SQuAD v1.1|F1: 89.4 (0.58)|
|GPT-2|INT8|WikiText-2 (raw)|perplexity: 20.9 (-1.17)|


<a name="tensorflow_models"></a>
## TensorFlow models

<a name="tensorflow_classification"></a>
### Classification

#### Quantization

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

#### Filter pruning

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|75.05|[resnet50_imagenet.json](configs/resnet50_imagenet.json)|-|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|74.96 (0.09)|[resnet50_imagenet_pruning_geometric_median.json](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion|ImageNet|75.09 (-0.04)|[resnet50_imagenet_pruning_geometric_median_int8.json](configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz)|

#### Accuracy-aware compressed training

|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**NNCF config file**|
| :---: | :---: | :---: | :---: | :---: |
|ResNet50|Sparsity 65% (magnitude)|ImageNet|74.37 (0.67)|[resnet50_imagenet_magnitude_sparsity_accuracy_aware.json](configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json)|



<a name="tensorflow_object_detection"></a>
### Object detection

#### Quantization

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RetinaNet|None|COCO 2017|33.43|[retinanet_coco.json](configs/retinanet_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz)|
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|33.12 (0.31)|[retinanet_coco_int8.json](configs/quantization/retinanet_coco_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_int8.tar.gz)|
|RetinaNet|Magnitude sparsity (50%)|COCO 2017|33.10 (0.33)|[retinanet_coco_magnitude_sparsity.json](configs/sparsity/retinanet_coco_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_magnitude_sparsity.tar.gz)|
|YOLO v4|None|COCO 2017|47.07|[yolo_v4_coco.json](configs/yolo_v4_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco.tar.gz)|
|YOLO v4|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|46.20 (0.87)|[yolo_v4_coco_int8.json](configs/quantization/yolo_v4_coco_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_int8.tar.gz)|
|YOLO v4|Magnitude sparsity, 50%|COCO 2017|46.49 (0.58)|[yolo_v4_coco_magnitude_sparsity.json](configs/sparsity/yolo_v4_coco_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_magnitude_sparsity.tar.gz)|

#### Filter pruning

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RetinaNet|None|COCO 2017|33.43|[retinanet_coco.json](configs/retinanet_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz)|
|RetinaNet|Filter pruning, 40%|COCO 2017|32.72 (0.71)|[retinanet_coco_pruning_geometric_median.json](configs/pruning/retinanet_coco_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median.tar.gz)|
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + filter pruning 40%|COCO 2017|32.67 (0.76)|[retinanet_coco_pruning_geometric_median_int8.json](configs/pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median_int8.tar.gz)|


<a name="tensorflow_instance_segmentation"></a>
### Instance segmentation

|Model|Compression algorithm|Dataset|            mAP (_drop_) %             |NNCF config file|Checkpoint|
| :---: | :---: | :---: |:-------------------------------------:| :---: | :---: |
|Mask-R-CNN|None|COCO 2017|        bbox: 37.33 segm: 33.56        |[mask_rcnn_coco.json](configs/mask_rcnn_coco.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz)|
|Mask-R-CNN|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017| bbox: 37.19 (0.14) segm: 33.54 (0.02) |[mask_rcnn_coco_int8.json](configs/quantization/mask_rcnn_coco_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_int8.tar.gz)|
|Mask-R-CNN|Magnitude sparsity, 50%|COCO 2017| bbox: 36.94 (0.39) segm: 33.23 (0.33) |[mask_rcnn_coco_magnitude_sparsity.json](configs/sparsity/mask_rcnn_coco_magnitude_sparsity.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_magnitude_sparsity.tar.gz)|


<a name="onnx_models"></a>
## ONNX models

<a name="onnx_classification"></a>
### Classification

|   ONNX Model    | Compression algorithm |Dataset|Accuracy (Drop) %|
| :---: |:---------------------:| :---: | :---: |
|ResNet-50| INT8 (Post-Training)  |ImageNet|74.63 (0.21)|
|ShuffleNet| INT8 (Post-Training)  |ImageNet|47.25 (0.18)|
|GoogleNet| INT8 (Post-Training)  |ImageNet|66.36 (0.3)|
|SqueezeNet V1.0| INT8 (Post-Training)  |ImageNet|54.3 (0.54)|
|MobileNet V2| INT8 (Post-Training)  |ImageNet|71.38 (0.49)|
|DenseNet-121| INT8 (Post-Training)  |ImageNet|60.16 (0.8)|
|VGG-16| INT8 (Post-Training)  |ImageNet|72.02 (0.0)|


<a name="onnx_object_detection"></a>
### Object detection

|ONNX Model| Compression algorithm | Dataset |mAP (drop) %|
| :---: |:---------------------:| :---: | :---: |
|SSD1200| INT8 (Post-Training)  |COCO2017|20.17 (0.17)|
|Tiny-YOLOv2| INT8 (Post-Training)  |VOC12|29.03 (0.23)|
