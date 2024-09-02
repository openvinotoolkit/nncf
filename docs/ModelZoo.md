# NNCF Compressed Model Zoo

Ready-to-use **Compressed LLMs** can be found on [OpenVINO Hugging Face page](https://huggingface.co/OpenVINO#models). Each model card includes NNCF parameters that were used to compress the model.

**INT8 Post-Training Quantization** ([PTQ](../README.md#post-training-quantization)) results for public Vision, NLP and GenAI models can be found on [OpenVino Performance Benchmarks page](https://docs.openvino.ai/2024/about-openvino/performance-benchmarks.html). PTQ results for ONNX models are available in the [ONNX](#onnx) section below.

**Quantization-Aware Training** ([QAT](../README.md#training-time-compression)) results for PyTorch and TensorFlow public models can be found below.

- [PyTorch](#pytorch)
  - [Classification](#pytorch-classification)
  - [Object Detection](#pytorch-object-detection)
  - [Semantic Segmentation](#pytorch-semantic-segmentation)
  - [Natural Language Processing (3rd-party training pipelines)](#pytorch-nlp-huggingface-transformers-powered-models)
- [TensorFlow](#tensorflow)
  - [Classification](#tensorflow-classification)
  - [Object Detection](#tensorflow-object-detection)
  - [Instance Segmentation](#tensorflow-instance-segmentation)
- [ONNX](#onnx)

## PyTorch

### PyTorch Classification

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">GoogLeNet</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>69.77</td>
      <td><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">GoogLeNet</td>
      <td align="left">• Filter pruning: 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.47 (0.30)</td>
      <td><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/googlenet_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>77.33</td>
      <td><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>77.45 (-0.12)</td>
      <td><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 61% (RB)</td>
      <td>ImageNet</td>
      <td>76.36 (0.97)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>71.87</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>71.07 (0.80)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>71.24 (0.63)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: Mixed, 58.88% INT8 / 41.12% INT4</td>
      <td>ImageNet</td>
      <td>70.95 (0.92)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_hawq.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int4_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 52% (RB)</td>
      <td>ImageNet</td>
      <td>71.09 (0.78)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Small)</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>67.66</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Small)</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>66.98 (0.68)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v3_small_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-18</td>
      <td align="left">• Filter pruning: 40%, magnitude criterion</td>
      <td>ImageNet</td>
      <td>69.27 (0.49)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_magnitude.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_magnitude.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-18</td>
      <td align="left">• Filter pruning: 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.31 (0.45)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-18</td>
      <td align="left">• Accuracy-aware compressed training<br>• Filter pruning: 60%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.2 (-0.6)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_accuracy_aware.json">Config</a></td>
      <td>-</td>
    </tr>
  <tr>
      <td align="left">ResNet-34</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>73.30</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">ResNet-34</td>
      <td align="left">• Filter pruning: 50%, geometric median criterion<br>• Knowledge distillation</td>
      <td>ImageNet</td>
      <td>73.11 (0.19)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet_pruning_geometric_median_kd.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet34_imagenet_pruning_geometric_median_kd.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>76.15</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>76.46 (-0.31)</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>76.39 (-0.24)</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: Mixed, 43.12% INT8 / 56.88% INT4</td>
      <td>ImageNet</td>
      <td>76.05 (0.10)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/resnet50_imagenet_mixed_int_hawq.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int4_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 61% (RB)</td>
      <td>ImageNet</td>
      <td>75.42 (0.73)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 50% (RB)</td>
      <td>ImageNet</td>
      <td>75.50 (0.65)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity50_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity50_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• Filter pruning: 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.57 (0.58)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• Accuracy-aware compressed training<br>• Filter pruning: 52.5%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.23 (0.93)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_accuracy_aware.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.1</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>58.19</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.1</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>58.22 (-0.03)</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.1</td>
      <td align="left">• QAT: INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>58.11 (0.08)</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.1</td>
      <td align="left">• QAT: Mixed,  52.83% INT8 / 47.17% INT4</td>
      <td>ImageNet</td>
      <td>57.57 (0.62)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq_old_eval.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int4_int8.pth">Download</a></td>
    </tr>
  </tbody>
</table>

### PyTorch Object Detection

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mAP (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">SSD300‑MobileNet</td>
      <td align="left">-</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>62.23</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD300‑MobileNet</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>62.95 (-0.72)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc_magnitude_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD300‑VGG‑BN</td>
      <td align="left">-</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>78.28</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD300‑VGG‑BN</td>
      <td align="left">• QAT: INT8</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>77.81 (0.47)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD300‑VGG‑BN</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>77.66 (0.62)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD300‑VGG‑BN</td>
      <td align="left">• Filter pruning: 40%, geometric median criterion</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>78.35 (-0.07)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth">Download</a></td>
    </tr>
  <tr>
      <td align="left">SSD512-VGG‑BN</td>
      <td align="left">-</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>80.26</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD512-VGG‑BN</td>
      <td align="left">• QAT: INT8</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>80.04 (0.22)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">SSD512-VGG‑BN</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>79.68 (0.58)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
  </tbody>
</table>

### PyTorch Semantic Segmentation

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mIoU (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">ICNet</td>
      <td align="left">-</td>
      <td>CamVid</td>
      <td>67.89</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ICNet</td>
      <td align="left">• QAT: INT8</td>
      <td>CamVid</td>
      <td>67.89 (0.00)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">ICNet</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 60% (Magnitude)</td>
      <td>CamVid</td>
      <td>67.16 (0.73)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">-</td>
      <td>CamVid</td>
      <td>71.95</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">• QAT: INT8</td>
      <td>CamVid</td>
      <td>71.89 (0.06)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 60% (Magnitude)</td>
      <td>CamVid</td>
      <td>72.46 (-0.51)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">-</td>
      <td>Mapillary</td>
      <td>56.24</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">• QAT: INT8</td>
      <td>Mapillary</td>
      <td>56.09 (0.15)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">• QAT: INT8<br>• Sparsity: 60% (Magnitude)</td>
      <td>Mapillary</td>
      <td>55.69 (0.55)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td align="left">UNet</td>
      <td align="left">• Filter pruning: 25%, geometric median criterion</td>
      <td>Mapillary</td>
      <td>55.64 (0.60)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_pruning_geometric_median.pth">Download</a></td>
    </tr>
  </tbody>
</table>

### PyTorch NLP (HuggingFace Transformers-powered models)

<table>
  <thead>
    <tr>
      <th>PyTorch Model</th>
      <th><img width="20" height="1">Compression algorithm<img width="20" height="1"></th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">BERT-base-cased</td>
      <td align="left">• QAT: INT8</td>
      <td>CoNLL2003</td>
      <td>99.18 (-0.01)</td>
    </tr>
    <tr>
      <td align="left">BERT-base-cased</td>
      <td align="left">• QAT: INT8</td>
      <td>MRPC</td>
      <td>84.8 (-0.24)</td>
    </tr>
    <tr>
      <td align="left">BERT-base-chinese</td>
      <td align="left">• QAT: INT8</td>
      <td>XNLI</td>
      <td>77.22 (0.46)</td>
    </tr>
    <tr>
      <td align="left">BERT-large<br>(Whole Word Masking)</td>
      <td align="left">• QAT: INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 92.68 (0.53)</td>
    </tr>
    <tr>
      <td align="left">DistilBERT-base</td>
      <td align="left">• QAT: INT8</td>
      <td>SST-2</td>
      <td>90.3 (0.8)</td>
    </tr>
    <tr>
      <td align="left">GPT-2</td>
      <td align="left">• QAT: INT8</td>
      <td>WikiText-2 (raw)</td>
      <td>perplexity: 20.9 (-1.17)</td>
    </tr>
  <tr>
      <td align="left">MobileBERT</td>
      <td align="left">• QAT: INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 89.4 (0.58)</td>
    </tr>
    <tr>
      <td align="left">RoBERTa-large</td>
      <td align="left">• QAT: INT8</td>
      <td>MNLI</td>
      <td>matched: 89.25 (1.35)</td>
    </tr>
  </tbody>
</table>

## TensorFlow

### TensorFlow Classification

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>77.91</td>
      <td><a href="../examples/tensorflow/classification/configs/inception_v3_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>78.39 (-0.48)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Sparsity: 61% (RB)</td>
      <td>ImageNet</td>
      <td>77.52 (0.39)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">Inception V3</td>
      <td align="left">• Sparsity: 54% (Magnitude)</td>
      <td>ImageNet</td>
      <td>77.86 (0.05)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>71.85</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v2_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>71.63 (0.22)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Sparsity: 52% (RB)</td>
      <td>ImageNet</td>
      <td>70.94 (0.91)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">• Sparsity: 50% (RB)</td>
      <td>ImageNet</td>
      <td>71.34 (0.51)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V2 (TensorFlow Hub MobileNet V2)</td>
      <td align="left">• Sparsity: 35% (Magnitude)</td>
      <td>ImageNet</td>
      <td>71.87 (-0.02)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_hub_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Large)</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>75.80</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v3_large_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Large)</td>
      <td align="left">• QAT: INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>75.04 (0.76)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_large_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Large)</td>
      <td align="left">• QAT: INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Sparsity: 42% (RB)</td>
      <td>ImageNet</td>
      <td>75.24 (0.56)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Small)</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>68.38</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v3_small_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Small)</td>
      <td align="left">• QAT: INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>67.79 (0.59)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">MobileNet V3 (Small)</td>
      <td align="left">• QAT: INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Sparsity: 42% (Magnitude)</td>
      <td>ImageNet</td>
      <td>67.44 (0.94)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">-</td>
      <td>ImageNet</td>
      <td>75.05</td>
      <td><a href="../examples/tensorflow/classification/configs/resnet50_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8</td>
      <td>ImageNet</td>
      <td>74.99 (0.06)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Sparsity: 65% (RB)</td>
      <td>ImageNet</td>
      <td>74.36 (0.69)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• Sparsity: 80% (RB)</td>
      <td>ImageNet</td>
      <td>74.38 (0.67)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/resnet50_imagenet_rb_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• Filter pruning: 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>74.96 (0.09)</td>
      <td><a href="../examples/tensorflow/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Filter pruning: 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.09 (-0.04)</td>
      <td><a href="../examples/tensorflow/classification/configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">ResNet50</td>
      <td align="left">• Accuracy-aware compressed training<br>• Sparsity: 65% (Magnitude)</td>
      <td>ImageNet</td>
      <td>74.37 (0.67)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json">Config</a></td>
      <td>-</td>
    </tr>
  </tbody>
</table>

### TensorFlow Object Detection

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mAP (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">RetinaNet</td>
      <td align="left">-</td>
      <td>COCO 2017</td>
      <td>33.43</td>
      <td><a href="../examples/tensorflow/object_detection/configs/retinanet_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">RetinaNet</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>33.12 (0.31)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/quantization/retinanet_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">RetinaNet</td>
      <td align="left">• Sparsity: 50% (Magnitude)</td>
      <td>COCO 2017</td>
      <td>33.10 (0.33)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/sparsity/retinanet_coco_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">RetinaNet</td>
      <td align="left">• Filter pruning: 40%</td>
      <td>COCO 2017</td>
      <td>32.72 (0.71)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/pruning/retinanet_coco_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">RetinaNet</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)<br>• Filter pruning: 40%</td>
      <td>COCO 2017</td>
      <td>32.67 (0.76)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median_int8.tar.gz">Download</a></td>
    </tr>
  <tr>
      <td align="left">YOLO v4</td>
      <td align="left">-</td>
      <td>COCO 2017</td>
      <td>47.07</td>
      <td><a href="../examples/tensorflow/object_detection/configs/yolo_v4_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">YOLO v4</td>
      <td align="left">• QAT: INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>46.20 (0.87)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/quantization/yolo_v4_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">YOLO v4</td>
      <td align="left">• Sparsity: 50% (Magnitude)</td>
      <td>COCO 2017</td>
      <td>46.49 (0.58)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/sparsity/yolo_v4_coco_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
  </tbody>
</table>

### TensorFlow Instance Segmentation

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mAP (<em>drop</em>) %</th>
      <th>Configuration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">Mask‑R‑CNN</td>
      <td align="left">-</td>
      <td>COCO 2017</td>
      <td>bbox: 37.33<br>segm: 33.56</td>
      <td><a href="../examples/tensorflow/segmentation/configs/mask_rcnn_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">Mask‑R‑CNN</td>
      <td align="left">• QAT: INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>bbox: 37.19 (0.14)<br>segm: 33.54 (0.02)</td>
      <td><a href="../examples/tensorflow/segmentation/configs/quantization/mask_rcnn_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td align="left">Mask‑R‑CNN</td>
      <td align="left">• Sparsity: 50% (Magnitude)</td>
      <td>COCO 2017</td>
      <td>bbox: 36.94 (0.39)<br>segm: 33.23 (0.33)</td>
      <td><a href="../examples/tensorflow/segmentation/configs/sparsity/mask_rcnn_coco_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
  </tbody>
</table>

## ONNX

### ONNX Classification

<table>
  <thead>
    <tr>
      <th>ONNX Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">DenseNet-121</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>60.16 (0.8)</td>
    </tr>
    <tr>
      <td align="left">GoogleNet</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>66.36 (0.3)</td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>71.38 (0.49)</td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>74.63 (0.21)</td>
    </tr>
    <tr>
      <td align="left">ShuffleNet</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>47.25 (0.18)</td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.0</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>54.3 (0.54)</td>
    </tr>
    <tr>
      <td align="left">VGG‑16</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>72.02 (0.0)</td>
    </tr>
  </tbody>
</table>

### ONNX Object Detection

<table>
  <thead>
    <tr>
      <th>ONNX Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mAP (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">SSD1200</td>
      <td align="left">PTQ</td>
      <td>COCO2017</td>
      <td>20.17 (0.17)</td>
    </tr>
    <tr>
      <td align="left">Tiny-YOLOv2</td>
      <td align="left">PTQ</td>
      <td>VOC12</td>
      <td>29.03 (0.23)</td>
    </tr>
  </tbody>
</table>
