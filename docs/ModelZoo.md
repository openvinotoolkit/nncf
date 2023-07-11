# NNCF Compressed Model Zoo

Here we present the results achieved using our sample scripts, example patches to third-party repositories and NNCF configuration files.

- [PyTorch](#pytorch)
  * [Classification](#pytorch-classification)
  * [Object Detection](#pytorch-object-detection)
  * [Semantic Segmentation](#pytorch-semantic-segmentation)
  * [Natural Language Processing (3rd-party training pipelines)](#pytorch-nlp-huggingface-transformers-powered-models)
- [TensorFlow](#tensorflow)
  * [Classification](#tensorflow-classification)
  * [Object Detection](#tensorflow-object-detection)
  * [Instance Segmentation](#tensorflow-instance-segmentation)
- [ONNX](#onnx)

## PyTorch

### PyTorch Classification

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>Accuracy&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>ResNet-50</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>76.15</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>76.46 (-0.31)</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>76.39 (-0.24)</td>
      <td><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization Mixed, 43.12% INT8 / 56.88% INT4</td>
      <td>ImageNet</td>
      <td>76.05 (0.10)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/resnet50_imagenet_mixed_int_hawq.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int4_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 + Sparsity 61% (RB)</td>
      <td>ImageNet</td>
      <td>75.42 (0.73)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 + Sparsity 50% (RB)</td>
      <td>ImageNet</td>
      <td>75.50 (0.65)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity50_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity50_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>77.33</td>
      <td><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>77.45 (-0.12)</td>
      <td><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>Quantization INT8 + Sparsity 61% (RB)</td>
      <td>ImageNet</td>
      <td>76.36 (0.97)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>71.87</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>71.07 (0.80)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>71.24 (0.63)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization Mixed, 58.88% INT8 / 41.12% INT4</td>
      <td>ImageNet</td>
      <td>70.95 (0.92)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_hawq.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int4_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8 + Sparsity 52% (RB)</td>
      <td>ImageNet</td>
      <td>71.09 (0.78)</td>
      <td><a href="../examples/torch/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_rb_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 small</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>67.66</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>MobileNet V3 small</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>66.98 (0.68)</td>
      <td><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v3_small_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SqueezeNet V1.1</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>58.19</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>SqueezeNet V1.1</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>58.22 (-0.03)</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SqueezeNet V1.1</td>
      <td>Quantization INT8 (per-tensor only)</td>
      <td>ImageNet</td>
      <td>58.11 (0.08)</td>
      <td><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8_per_tensor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8_per_tensor.pth">Download</a></td>
    </tr>
    <tr>
      <td>SqueezeNet V1.1</td>
      <td>Quantization Mixed, 52.83% INT8 / 47.17% INT4</td>
      <td>ImageNet</td>
      <td>57.57 (0.62)</td>
      <td><a href="../examples/torch/classification/configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq_old_eval.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int4_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>69.76</td>
      <td><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet-34</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>73.30</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>GoogLeNet</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>69.77</td>
      <td><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>Binarization XNOR (weights), scale/threshold (activations)</td>
      <td>ImageNet</td>
      <td>61.67 (8.09)</td>
      <td><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_xnor.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_xnor.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>Binarization DoReFa (weights), scale/threshold (activations)</td>
      <td>ImageNet</td>
      <td>61.63 (8.13)</td>
      <td><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_dorefa.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_dorefa.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Filter pruning, 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.57 (0.58)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>Filter pruning, 40%, magnitude criterion</td>
      <td>ImageNet</td>
      <td>69.27 (0.49)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_magnitude.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_magnitude.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>Filter pruning, 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.31 (0.45)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-34</td>
      <td>Filter pruning, 50%, geometric median criterion + KD</td>
      <td>ImageNet</td>
      <td>73.11 (0.19)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet_pruning_geometric_median_kd.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet34_imagenet_pruning_geometric_median_kd.pth">Download</a></td>
    </tr>
    <tr>
      <td>GoogLeNet</td>
      <td>Filter pruning, 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.47 (0.30)</td>
      <td><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/googlenet_imagenet_pruning_geometric_median.pth">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Accuracy-aware compressed training, Filter pruning, 52.5%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.23 (0.93)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_accuracy_aware.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet-18</td>
      <td>Accuracy-aware compressed training, Filter pruning, 60%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>69.2 (-0.6)</td>
      <td><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_accuracy_aware.json">Config</a></td>
      <td>-</td>
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
      <th>mAP&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>SSD300-MobileNet</td>
      <td>None</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>62.23</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD300-MobileNet</td>
      <td>Quantization INT8 + Sparsity 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>62.95 (-0.72)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc_magnitude_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD300-VGG-BN</td>
      <td>None</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>78.28</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD300-VGG-BN</td>
      <td>Quantization INT8</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>77.81 (0.47)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD300-VGG-BN</td>
      <td>Quantization INT8 + Sparsity 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>77.66 (0.62)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD512-VGG-BN</td>
      <td>None</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>80.26</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD512-VGG-BN</td>
      <td>Quantization INT8</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>80.04 (0.22)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD512-VGG-BN</td>
      <td>Quantization INT8 + Sparsity 70% (Magnitude)</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>79.68 (0.58)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>SSD300-VGG-BN</td>
      <td>Filter pruning, 40%, geometric median criterion</td>
      <td>VOC12+07 train, VOC07 eval</td>
      <td>78.35 (-0.07)</td>
      <td><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth">Download</a></td>
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
      <th>mIoU&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>UNet</td>
      <td>None</td>
      <td>CamVid</td>
      <td>71.95</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>Quantization INT8</td>
      <td>CamVid</td>
      <td>71.89 (0.06)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>Quantization INT8 + Sparsity 60% (Magnitude)</td>
      <td>CamVid</td>
      <td>72.46 (-0.51)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ICNet</td>
      <td>None</td>
      <td>CamVid</td>
      <td>67.89</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid.pth">Download</a></td>
    </tr>
    <tr>
      <td>ICNet</td>
      <td>Quantization INT8</td>
      <td>CamVid</td>
      <td>67.89 (0.00)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>ICNet</td>
      <td>Quantization INT8 + Sparsity 60% (Magnitude)</td>
      <td>CamVid</td>
      <td>67.16 (0.73)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>None</td>
      <td>Mapillary</td>
      <td>56.24</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>Quantization INT8</td>
      <td>Mapillary</td>
      <td>56.09 (0.15)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>Quantization INT8 + Sparsity 60% (Magnitude)</td>
      <td>Mapillary</td>
      <td>55.69 (0.55)</td>
      <td><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_magnitude_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_magnitude_sparsity_int8.pth">Download</a></td>
    </tr>
    <tr>
      <td>UNet</td>
      <td>Filter pruning, 25%, geometric median criterion</td>
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
      <th>Accuracy&nbsp(<em>drop</em>)&nbsp%</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>BERT-base-chinese</td>
      <td>Quantization INT8</td>
      <td>XNLI</td>
      <td>77.22 (0.46)</td>
    </tr>
    <tr>
      <td>BERT-base-cased</td>
      <td>Quantization INT8</td>
      <td>CoNLL2003</td>
      <td>99.18 (-0.01)</td>
    </tr>
    <tr>
      <td>BERT-base-cased</td>
      <td>Quantization INT8</td>
      <td>MRPC</td>
      <td>84.8 (-0.24)</td>
    </tr>
    <tr>
      <td>BERT-large (Whole Word Masking)</td>
      <td>Quantization INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 92.68 (0.53)</td>
    </tr>
    <tr>
      <td>RoBERTa-large</td>
      <td>Quantization INT8</td>
      <td>MNLI</td>
      <td>matched: 89.25 (1.35)</td>
    </tr>
    <tr>
      <td>DistilBERT-base</td>
      <td>Quantization INT8</td>
      <td>SST-2</td>
      <td>90.3 (0.8)</td>
    </tr>
    <tr>
      <td>MobileBERT</td>
      <td>Quantization INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 89.4 (0.58)</td>
    </tr>
    <tr>
      <td>GPT-2</td>
      <td>Quantization INT8</td>
      <td>WikiText-2 (raw)</td>
      <td>perplexity: 20.9 (-1.17)</td>
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
      <th>Accuracy&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>Inception V3</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>77.91</td>
      <td><a href="../examples/tensorflow/classification/configs/inception_v3_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>78.39 (-0.48)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 61% (RB)</td>
      <td>ImageNet</td>
      <td>77.52 (0.39)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>Inception V3</td>
      <td>Sparsity 54% (Magnitude)</td>
      <td>ImageNet</td>
      <td>77.86 (0.05)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>71.85</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v2_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>71.63 (0.22)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 52% (RB)</td>
      <td>ImageNet</td>
      <td>70.94 (0.91)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Sparsity 50% (RB)</td>
      <td>ImageNet</td>
      <td>71.34 (0.51)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V2 (TensorFlow Hub MobileNet V2)</td>
      <td>Sparsity 35% (Magnitude)</td>
      <td>ImageNet</td>
      <td>71.87 (-0.02)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_hub_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 (Small)</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>68.38</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v3_small_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>MobileNet V3 (Small)</td>
      <td>Quantization INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>67.79 (0.59)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 (Small)</td>
      <td>Quantization INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (Magnitude)</td>
      <td>ImageNet</td>
      <td>67.44 (0.94)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 (Large)</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>75.80</td>
      <td><a href="../examples/tensorflow/classification/configs/mobilenet_v3_large_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>MobileNet V3 (Large)</td>
      <td>Quantization INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>ImageNet</td>
      <td>75.04 (0.76)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_large_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 (Large)</td>
      <td>Quantization INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (RB)</td>
      <td>ImageNet</td>
      <td>75.24 (0.56)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>None</td>
      <td>ImageNet</td>
      <td>75.05</td>
      <td><a href="../examples/tensorflow/classification/configs/resnet50_imagenet.json">Config</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8</td>
      <td>ImageNet</td>
      <td>74.99 (0.06)</td>
      <td><a href="../examples/tensorflow/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 65% (RB)</td>
      <td>ImageNet</td>
      <td>74.36 (0.69)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Sparsity 80% (RB)</td>
      <td>ImageNet</td>
      <td>74.38 (0.67)</td>
      <td><a href="../examples/tensorflow/classification/configs/sparsity/resnet50_imagenet_rb_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Filter pruning, 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>74.96 (0.09)</td>
      <td><a href="../examples/tensorflow/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion</td>
      <td>ImageNet</td>
      <td>75.09 (-0.04)</td>
      <td><a href="../examples/tensorflow/classification/configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>ResNet50</td>
      <td>Accuracy-aware compressed training, Sparsity 65% (magnitude)</td>
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
      <th>mAP&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>RetinaNet</td>
      <td>None</td>
      <td>COCO 2017</td>
      <td>33.43</td>
      <td><a href="../examples/tensorflow/object_detection/configs/retinanet_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>33.12 (0.31)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/quantization/retinanet_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>Magnitude sparsity (50%)</td>
      <td>COCO 2017</td>
      <td>33.10 (0.33)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/sparsity/retinanet_coco_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>YOLO v4</td>
      <td>None</td>
      <td>COCO 2017</td>
      <td>47.07</td>
      <td><a href="../examples/tensorflow/object_detection/configs/yolo_v4_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>YOLO v4</td>
      <td>Quantization INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>46.20 (0.87)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/quantization/yolo_v4_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>YOLO v4</td>
      <td>Magnitude sparsity, 50%</td>
      <td>COCO 2017</td>
      <td>46.49 (0.58)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/sparsity/yolo_v4_coco_magnitude_sparsity.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_magnitude_sparsity.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>None</td>
      <td>COCO 2017</td>
      <td>33.43</td>
      <td><a href="../examples/tensorflow/object_detection/configs/retinanet_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>Filter pruning, 40%</td>
      <td>COCO 2017</td>
      <td>32.72 (0.71)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/pruning/retinanet_coco_pruning_geometric_median.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>RetinaNet</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + filter pruning 40%</td>
      <td>COCO 2017</td>
      <td>32.67 (0.76)</td>
      <td><a href="../examples/tensorflow/object_detection/configs/pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median_int8.tar.gz">Download</a></td>
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
      <th>mAP&nbsp(<em>drop</em>)&nbsp%</th>
      <th>NNCF config file</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>Mask-R-CNN</td>
      <td>None</td>
      <td>COCO 2017</td>
      <td>bbox: 37.33 segm: 33.56</td>
      <td><a href="../examples/tensorflow/segmentation/configs/mask_rcnn_coco.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>Mask-R-CNN</td>
      <td>Quantization INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
      <td>COCO 2017</td>
      <td>bbox: 37.19 (0.14) segm: 33.54 (0.02)</td>
      <td><a href="../examples/tensorflow/segmentation/configs/quantization/mask_rcnn_coco_int8.json">Config</a></td>
      <td><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_int8.tar.gz">Download</a></td>
    </tr>
    <tr>
      <td>Mask-R-CNN</td>
      <td>Magnitude sparsity, 50%</td>
      <td>COCO 2017</td>
      <td>bbox: 36.94 (0.39) segm: 33.23 (0.33)</td>
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
      <th>Accuracy&nbsp(<em>drop</em>)&nbsp%</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>ResNet-50</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>74.63 (0.21)</td>
    </tr>
    <tr>
      <td>ShuffleNet</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>47.25 (0.18)</td>
    </tr>
    <tr>
      <td>GoogleNet</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>66.36 (0.3)</td>
    </tr>
    <tr>
      <td>SqueezeNet V1.0</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>54.3 (0.54)</td>
    </tr>
    <tr>
      <td>MobileNet V2</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>71.38 (0.49)</td>
    </tr>
    <tr>
      <td>DenseNet-121</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>ImageNet</td>
      <td>60.16 (0.8)</td>
    </tr>
    <tr>
      <td>VGG-16</td>
      <td>Quantization INT8 (Post-Training)</td>
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
      <th>mAP&nbsp(<em>drop</em>)&nbsp%</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>SSD1200</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>COCO2017</td>
      <td>20.17 (0.17)</td>
    </tr>
    <tr>
      <td>Tiny-YOLOv2</td>
      <td>Quantization INT8 (Post-Training)</td>
      <td>VOC12</td>
      <td>29.03 (0.23)</td>
    </tr>
  </tbody>
</table>

