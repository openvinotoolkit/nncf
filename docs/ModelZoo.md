# NNCF Compressed Model Zoo

Here we present the results achieved using our sample scripts, example patches to third-party repositories and NNCF configuration files.

See [sample scripts](../README.md#model-compression-tutorials-and-samples) and [example patches](../README.md#git-patches-for-third-party-repository)
to find instruction and links to exact configuration files and final checkpoints.
- [PyTorch](#pytorch_models)
  * [Classification](#pytorch_classification)
  * [Object Detection](#pytorch_object_detection)
  * [Semantic Segmentation](#pytorch_semantic_segmentation)
  * [Natural Language Processing (3rd-party training pipelines)](#pytorch_nlp)
- [TensorFlow](#tensorflow_models)
  * [Classification](#tensorflow_classification)
  * [Object Detection](#tensorflow_object_detection)
  * [Instance Segmentation](#tensorflow_instance_segmentation)
- [ONNX](#onnx_models)

<a id="pytorch_models"></a>
## PyTorch

<a id="pytorch_classification"></a>
### PyTorch Classification

<table>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">Accuracy (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.15</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.46 (-0.31)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.39 (-0.24)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/resnet50_imagenet_int8_per_tensor.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Mixed, 43.12% INT8 / 56.88% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.05 (0.10)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/resnet50_imagenet_mixed_int_hawq.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 + Sparsity 61% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.42 (0.73)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 + Sparsity 50% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.50 (0.65)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity50_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity50_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.33</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.45 (-0.12)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8 + Sparsity 61% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.36 (0.97)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.87</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.07 (0.80)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.24 (0.63)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/mobilenet_v2_imagenet_int8_per_tensor.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">Mixed, 58.88% INT8 / 41.12% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">70.95 (0.92)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_hawq.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 + Sparsity 52% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.09 (0.78)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 small</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">67.66</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 small</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">66.98 (0.68)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v3_small_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.19</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.22 (-0.03)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.11 (0.08)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/squeezenet1_1_imagenet_int8_per_tensor.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">Mixed, 52.83% INT8 / 47.17% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">57.57 (0.62)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq_old_eval.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.76</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-34</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">73.30</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogLeNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.77</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Binarization<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">XNOR (weights), scale/threshold (activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">61.67 (8.09)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_xnor.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_xnor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">DoReFa (weights), scale/threshold (activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">61.63 (8.13)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/resnet18_imagenet_binarization_dorefa.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_dorefa.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.57 (0.58)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">Filter pruning, 40%, magnitude criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.27 (0.49)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_magnitude.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_magnitude.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.31 (0.45)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-34</td>
			<td style="text-align: center;">Filter pruning, 50%, geometric median criterion + KD</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">73.11 (0.19)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet34_imagenet_pruning_geometric_median_kd.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet34_imagenet_pruning_geometric_median_kd.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogLeNet</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.47 (0.30)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/googlenet_imagenet_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/googlenet_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Accuracy-aware compressed training<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Filter pruning, 52.5%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.23 (0.93)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet50_imagenet_pruning_accuracy_aware.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">Filter pruning, 60%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.2 (-0.6)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/resnet18_imagenet_pruning_accuracy_aware.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
	</tbody>
</table>


<a id="pytorch_object_detection"></a>
### PyTorch Object Detection

<table>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">mAP (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">SSD300-MobileNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">62.23</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-MobileNet</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">62.95 (-0.72)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_mobilenet_voc_magnitude_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">78.28</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">77.81 (0.47)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">77.66 (0.62)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">80.26</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">80.04 (0.22)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">79.68 (0.58)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd512_vgg_voc_magnitude_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">78.35 (-0.07)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/ssd300_vgg_voc_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
</table>


<a id="pytorch_semantic_segmentation"></a>
### PyTorch Semantic Segmentation

<table>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">mIoU (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">71.95</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_camvid.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">71.89 (0.06)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">72.46 (-0.51)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_camvid_magnitude_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.89</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.89 (0.00)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.16 (0.73)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/icnet_camvid_magnitude_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">56.24</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">56.09 (0.15)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">55.69 (0.55)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_magnitude_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">Filter pruning, 25%, geometric median criterion</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">55.64 (0.60)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/unet_mapillary_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
</table>

<a id="pytorch_nlp"></a>
### PyTorch NLP (HuggingFace Transformers-powered models)

<table>
	<thead>
		<tr>
			<th style="text-align: center;">PyTorch Model</th>
			<th style="text-align: center;"><img width="20" height="1">Compression algorithm<img width="20" height="1"></th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">Accuracy (Drop) %</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">BERT-base-chinese</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">XNLI</td>
			<td style="text-align: center;">77.22 (0.46)</td>
		</tr>
		<tr>
			<td style="text-align: center;">BERT-base-cased</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">CoNLL2003</td>
			<td style="text-align: center;">99.18 (-0.01)</td>
		</tr>
		<tr>
			<td style="text-align: center;">BERT-base-cased</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">MRPC</td>
			<td style="text-align: center;">84.8 (-0.24)</td>
		</tr>
		<tr>
			<td style="text-align: center;">BERT-large (Whole Word Masking)</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">SQuAD v1.1</td>
			<td style="text-align: center;">F1: 92.68 (0.53)</td>
		</tr>
		<tr>
			<td style="text-align: center;">RoBERTa-large</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">MNLI</td>
			<td style="text-align: center;">matched: 89.25 (1.35)</td>
		</tr>
		<tr>
			<td style="text-align: center;">DistilBERT-base</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">SST-2</td>
			<td style="text-align: center;">90.3 (0.8)</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileBERT</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">SQuAD v1.1</td>
			<td style="text-align: center;">F1: 89.4 (0.58)</td>
		</tr>
		<tr>
			<td style="text-align: center;">GPT-2</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">WikiText-2 (raw)</td>
			<td style="text-align: center;">perplexity: 20.9 (-1.17)</td>
		</tr>
	</tbody>
</table>


<a id="tensorflow_models"></a>
## TensorFlow

<a id="tensorflow_classification"></a>
### TensorFlow Classification

<table>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">Accuracy (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.91</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/inception_v3_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">78.39 (-0.48)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/quantization/inception_v3_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 61% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.52 (0.39)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">Sparsity 54% (Magnitude)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.86 (0.05)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.85</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/mobilenet_v2_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.63 (0.22)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v2_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 52% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">70.94 (0.91)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">Sparsity 50% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.34 (0.51)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2 (TensorFlow Hub MobileNet V2)</td>
			<td style="text-align: center;">Sparsity 35% (Magnitude)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.87 (-0.02)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_hub_imagenet_magnitude_sparsity.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Small)</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">68.38</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/mobilenet_v3_small_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Small)</td>
			<td style="text-align: center;">INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">67.79 (0.59)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_small_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Small)</td>
			<td style="text-align: center;">INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (Magnitude)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">67.44 (0.94)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Large)</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.80</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/mobilenet_v3_large_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Large)</td>
			<td style="text-align: center;">INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.04 (0.76)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/quantization/mobilenet_v3_large_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 (Large)</td>
			<td style="text-align: center;">INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.24 (0.56)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.05</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/resnet50_imagenet.json">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.99 (0.06)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/quantization/resnet50_imagenet_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 65% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.36 (0.69)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Sparsity 80% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.38 (0.67)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity/resnet50_imagenet_rb_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity.tar.gz">Download</a></td>
		</tr>
	</tbody>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.96 (0.09)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/pruning/resnet50_imagenet_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.09 (-0.04)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz">Download</a></td>
		</tr>
	</tbody>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Accuracy-aware compressed training<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet50</td>
			<td style="text-align: center;">Sparsity 65% (magnitude)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.37 (0.67)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/classification/configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json">Config</a></td>
            <td style="text-align: center;">-</td>
		</tr>
	</tbody>
</table>


<a id="tensorflow_object_detection"></a>
### TensorFlow Object Detection

<table>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">mAP (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">33.43</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/retinanet_coco.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">33.12 (0.31)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/quantization/retinanet_coco_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">Magnitude sparsity (50%)</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">33.10 (0.33)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/sparsity/retinanet_coco_magnitude_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_magnitude_sparsity.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">YOLO v4</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">47.07</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/yolo_v4_coco.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">YOLO v4</td>
			<td style="text-align: center;">INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">46.20 (0.87)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/quantization/yolo_v4_coco_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">YOLO v4</td>
			<td style="text-align: center;">Magnitude sparsity, 50%</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">46.49 (0.58)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/sparsity/yolo_v4_coco_magnitude_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_magnitude_sparsity.tar.gz">Download</a></td>
		</tr>
	</tbody>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">33.43</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/retinanet_coco.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">Filter pruning, 40%</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">32.72 (0.71)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/pruning/retinanet_coco_pruning_geometric_median.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">RetinaNet</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + filter pruning 40%</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">32.67 (0.76)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/object_detection/configs/pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median_int8.tar.gz">Download</a></td>
		</tr>
	</tbody>
</table>

<a id="tensorflow_instance_segmentation"></a>
### TensorFlow Instance Segmentation

<table>
	<thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Quantization<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">mAP (<em>drop</em>) %</th>
			<th style="text-align: center;">NNCF config file</th>
			<th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">Mask-R-CNN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">bbox: 37.33 segm: 33.56</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/instance_segmentation/configs/mask_rcnn_coco.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Mask-R-CNN</td>
			<td style="text-align: center;">INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">bbox: 37.19 (0.14) segm: 33.54 (0.02)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/instance_segmentation/configs/quantization/mask_rcnn_coco_int8.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_int8.tar.gz">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Mask-R-CNN</td>
			<td style="text-align: center;">Magnitude sparsity, 50%</td>
			<td style="text-align: center;">COCO 2017</td>
			<td style="text-align: center;">bbox: 36.94 (0.39) segm: 33.23 (0.33)</td>
			<td style="text-align: center;"><a href="../examples/tensorflow/instance_segmentation/configs/sparsity/mask_rcnn_coco_magnitude_sparsity.json">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_magnitude_sparsity.tar.gz">Download</a></td>
		</tr>
	</tbody>
</table>


<a id="onnx_models"></a>
## ONNX

<a id="onnx_classification"></a>
### ONNX Classification

<table>
	<thead>
		<tr>
			<th style="text-align: center;">ONNX Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">Accuracy (Drop) %</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">74.63 (0.21)</td>
		</tr>
		<tr>
			<td style="text-align: center;">ShuffleNet</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">47.25 (0.18)</td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogleNet</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">66.36 (0.3)</td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.0</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">54.3 (0.54)</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.38 (0.49)</td>
		</tr>
		<tr>
			<td style="text-align: center;">DenseNet-121</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">60.16 (0.8)</td>
		</tr>
		<tr>
			<td style="text-align: center;">VGG-16</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">72.02 (0.0)</td>
		</tr>
	</tbody>
</table>



<a id="onnx_object_detection"></a>
### ONNX Object Detection

<table>
	<thead>
		<tr>
			<th style="text-align: center;">ONNX Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">mAP (drop) %</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">SSD1200</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">COCO2017</td>
			<td style="text-align: center;">20.17 (0.17)</td>
		</tr>
		<tr>
			<td style="text-align: center;">Tiny-YOLOv2</td>
			<td style="text-align: center;">INT8 (Post-Training)</td>
			<td style="text-align: center;">VOC12</td>
			<td style="text-align: center;">29.03 (0.23)</td>
		</tr>
	</tbody>
</table>

