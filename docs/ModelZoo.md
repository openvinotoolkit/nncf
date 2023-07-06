# NNCF Compressed Model Zoo

Results achieved using sample scripts, example patches to third-party repositories and NNCF configuration files provided
with this repository. See README.md files for [sample scripts](#model-compression-samples) and [example patches](#third-party-repository-integration)
to find instruction and links to exact configuration files and final checkpoints.
- [PyTorch models](#pytorch_models)
  * [Classification](#pytorch_classification)
  * [Object detection](#pytorch_object_detection)
  * [Semantic segmentation](#pytorch_semantic_segmentation)
  * [Natural language processing (3rd-party training pipelines)](#pytorch_nlp)
- [TensorFlow models](#tensorflow_models)
  * [Classification](#tensorflow_classification)
  * [Object detection](#tensorflow_object_detection)
  * [Instance segmentation](#tensorflow_instance_segmentation)
- [ONNX models](#onnx_models)

<a id="pytorch_models"></a>
## PyTorch

<a id="pytorch_classification"></a>
### Classification

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
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.46 (-0.31)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.39 (-0.24)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Mixed, 43.12% INT8 / 56.88% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.05 (0.10)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 + Sparsity 61% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.42 (0.73)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">INT8 + Sparsity 50% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.50 (0.65)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_rb_sparsity50_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.33</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">77.45 (-0.12)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">Inception V3</td>
			<td style="text-align: center;">INT8 + Sparsity 61% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.36 (0.97)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/inception_v3_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.87</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.07 (0.80)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.24 (0.63)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">Mixed, 58.88% INT8 / 41.12% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">70.95 (0.92)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V2</td>
			<td style="text-align: center;">INT8 + Sparsity 52% (RB)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">71.09 (0.78)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/sparsity_quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v2_imagenet_rb_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 small</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">67.66</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">MobileNet V3 small</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">66.98 (0.68)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/mobilenet_v3_small_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.19</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.22 (-0.03)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">INT8 (per-tensor only)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">58.11 (0.08)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int8_per_tensor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SqueezeNet V1.1</td>
			<td style="text-align: center;">Mixed, 52.83% INT8 / 47.17% INT4</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">57.57 (0.62)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/mixed_precision/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/squeezenet1_1_imagenet_int4_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.76</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-34</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">73.30</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogLeNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.77</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Binarization<br>&nbsp</th>
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
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.76</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">XNOR (weights), scale/threshold (activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">61.67 (8.09)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_xnor.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">DoReFa (weights), scale/threshold (activations)</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">61.63 (8.13)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_binarization_dorefa.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
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
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">75.57 (0.58)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet50_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.76</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">Filter pruning, 40%, magnitude criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.27 (0.49)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_magnitude.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-18</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.31 (0.45)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet18_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-34</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">73.30</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">ResNet-34</td>
			<td style="text-align: center;">Filter pruning, 50%, geometric median criterion + KD</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">73.11 (0.19)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/resnet34_imagenet_pruning_geometric_median_kd.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogLeNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.77</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
		<tr>
			<td style="text-align: center;">GoogLeNet</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.47 (0.30)</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/pruning/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/googlenet_imagenet_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Accuracy-aware compressed training<br>&nbsp</th>
		</tr>
		<tr>
			<th style="text-align: center;">Model</th>
			<th style="text-align: center;">Compression algorithm</th>
			<th style="text-align: center;">Dataset</th>
			<th style="text-align: center;">Accuracy (Drop) %</th>
			<th style="text-align: center;">NNCF config file</th>
            <th style="text-align: center;">Checkpoint</th>
		</tr>
	</thead>
	<tbody align="center">
		<tr>
			<td style="text-align: center;">ResNet-50</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">76.16</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/quantization/Config">Config</a></td>
			<td style="text-align: center;">-</td>
		</tr>
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
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">ImageNet</td>
			<td style="text-align: center;">69.8</td>
			<td style="text-align: center;"><a href="../examples/torch/classification/configs/binarization/Config">Config</a></td>
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
### Object detection

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
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-MobileNet</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">62.95 (-0.72)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_mobilenet_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">78.28</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">77.81 (0.47)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">77.66 (0.62)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">80.26</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">80.04 (0.22)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD512-VGG-BN</td>
			<td style="text-align: center;">INT8 + Sparsity 70% (Magnitude)</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">79.68 (0.58)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd512_vgg_voc_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
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
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">78.28</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">SSD300-VGG-BN</td>
			<td style="text-align: center;">Filter pruning, 40%, geometric median criterion</td>
			<td style="text-align: center;">VOC12+07 train, VOC07 eval</td>
			<td style="text-align: center;">78.35 (-0.07)</td>
			<td style="text-align: center;"><a href="../examples/torch/object_detection/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/ssd300_vgg_voc_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
</table>


<a id="pytorch_semantic_segmentation"></a>
### Semantic segmentation

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
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">71.89 (0.06)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">72.46 (-0.51)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.89</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.89 (0.00)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">ICNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">CamVid</td>
			<td style="text-align: center;">67.16 (0.73)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/icnet_camvid_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">None</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">56.24</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">56.09 (0.15)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_int8.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">INT8 + Sparsity 60% (Magnitude)</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">55.69 (0.55)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_magnitude_sparsity_int8.pth">Download</a></td>
		</tr>
	</tbody>
    <thead>
		<tr>
			<th style="text-align: center;" colspan="6"><br>Filter pruning<br>&nbsp</th>
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
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">56.24</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary.pth">Download</a></td>
		</tr>
		<tr>
			<td style="text-align: center;">UNet</td>
			<td style="text-align: center;">Filter pruning, 25%, geometric median criterion</td>
			<td style="text-align: center;">Mapillary</td>
			<td style="text-align: center;">55.64 (0.60)</td>
			<td style="text-align: center;"><a href="../examples/torch/semantic_segmentation/configs/Config">Config</a></td>
			<td style="text-align: center;"><a href="https://storage.openvinotoolkit.org/repositories/nncf/models/develop/torch/unet_mapillary_pruning_geometric_median.pth">Download</a></td>
		</tr>
	</tbody>
</table>

<a id="pytorch_nlp"></a>
### NLP (HuggingFace Transformers-powered models)

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
## TensorFlow models

<a id="tensorflow_classification"></a>
### Classification

#### Quantization

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Inception V3|None|ImageNet|77.91|[inception_v3_imagenet.json](configs/inception_v3_imagenet.json)|-|
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|78.39 (-0.48)|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_int8.tar.gz)|
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 61% (RB)|ImageNet|77.52 (0.39)|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_rb_sparsity_int8.tar.gz)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.86 (0.05)|[inception_v3_imagenet_magnitude_sparsity.json](configs/sparsity/inception_v3_imagenet_magnitude_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/inception_v3_imagenet_magnitude_sparsity.tar.gz)|
|MobileNet V2|None|ImageNet|71.85|[mobilenet_v2_imagenet.json](configs/mobilenet_v2_imagenet.json)|-|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|71.63 (0.22)|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_int8.tar.gz)|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 52% (RB)|ImageNet|70.94 (0.91)|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity_int8.tar.gz)|
|MobileNet V2| Sparsity 50% (RB)|ImageNet|71.34 (0.51)|[mobilenet_v2_imagenet_rb_sparsity.json](configs/sparsity/mobilenet_v2_imagenet_rb_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_imagenet_rb_sparsity.tar.gz)|
|MobileNet V2 (TensorFlow Hub MobileNet V2)|Sparsity 35% (Magnitude)|ImageNet|71.87 (-0.02)|[mobilenet_v2_hub_imagenet_magnitude_sparsity.json](configs/sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v2_hub_imagenet_magnitude_sparsity.tar.gz)|
|MobileNet V3 (Small)|None|ImageNet|68.38|[mobilenet_v3_small_imagenet.json](configs/mobilenet_v3_small_imagenet.json)|-|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|67.79 (0.59)|[mobilenet_v3_small_imagenet_int8.json](configs/quantization/mobilenet_v3_small_imagenet_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_int8.tar.gz)|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (Magnitude)|ImageNet|67.44 (0.94)|[mobilenet_v3_small_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_small_imagenet_rb_sparsity_int8.tar.gz)|
|MobileNet V3 (Large)|None|ImageNet|75.80|[mobilenet_v3_large_imagenet.json](configs/mobilenet_v3_large_imagenet.json)|-|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|75.04 (0.76)|[mobilenet_v3_large_imagenet_int8.json](configs/quantization/mobilenet_v3_large_imagenet_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_int8.tar.gz)|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (RB)|ImageNet|75.24 (0.56)|[mobilenet_v3_large_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mobilenet_v3_large_imagenet_rb_sparsity_int8.tar.gz)|
|ResNet-50|None|ImageNet|75.05|[resnet50_imagenet.json](configs/resnet50_imagenet.json)|-|
|ResNet-50|INT8|ImageNet|74.99 (0.06)|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_int8.tar.gz)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 65% (RB)|ImageNet|74.36 (0.69)|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity_int8.tar.gz)|
|ResNet-50|Sparsity 80% (RB)|ImageNet|74.38 (0.67)|[resnet50_imagenet_rb_sparsity.json](configs/sparsity/resnet50_imagenet_rb_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_rb_sparsity.tar.gz)|

#### Filter pruning

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|75.05|[resnet50_imagenet.json](configs/resnet50_imagenet.json)|-|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|74.96 (0.09)|[resnet50_imagenet_pruning_geometric_median.json](configs/pruning/resnet50_imagenet_pruning_geometric_median.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median.tar.gz)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion|ImageNet|75.09 (-0.04)|[resnet50_imagenet_pruning_geometric_median_int8.json](configs/pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/resnet50_imagenet_pruning_geometric_median_int8.tar.gz)|

#### Accuracy-aware compressed training

|**Model**|**Compression algorithm**|**Dataset**|**Accuracy (Drop) %**|**NNCF config file**|
| :---: | :---: | :---: | :---: | :---: |
|ResNet50|Sparsity 65% (magnitude)|ImageNet|74.37 (0.67)|[resnet50_imagenet_magnitude_sparsity_accuracy_aware.json](configs/sparsity/resnet50_imagenet_magnitude_sparsity_accuracy_aware.json)|



<a id="tensorflow_object_detection"></a>
### Object detection

#### Quantization

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RetinaNet|None|COCO 2017|33.43|[retinanet_coco.json](configs/retinanet_coco.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz)|
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|33.12 (0.31)|[retinanet_coco_int8.json](configs/quantization/retinanet_coco_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_int8.tar.gz)|
|RetinaNet|Magnitude sparsity (50%)|COCO 2017|33.10 (0.33)|[retinanet_coco_magnitude_sparsity.json](configs/sparsity/retinanet_coco_magnitude_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_magnitude_sparsity.tar.gz)|
|YOLO v4|None|COCO 2017|47.07|[yolo_v4_coco.json](configs/yolo_v4_coco.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco.tar.gz)|
|YOLO v4|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|46.20 (0.87)|[yolo_v4_coco_int8.json](configs/quantization/yolo_v4_coco_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_int8.tar.gz)|
|YOLO v4|Magnitude sparsity, 50%|COCO 2017|46.49 (0.58)|[yolo_v4_coco_magnitude_sparsity.json](configs/sparsity/yolo_v4_coco_magnitude_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/yolo_v4_coco_magnitude_sparsity.tar.gz)|

#### Filter pruning

|Model|Compression algorithm|Dataset|mAP (_drop_) %|NNCF config file|Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|RetinaNet|None|COCO 2017|33.43|[retinanet_coco.json](configs/retinanet_coco.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco.tar.gz)|
|RetinaNet|Filter pruning, 40%|COCO 2017|32.72 (0.71)|[retinanet_coco_pruning_geometric_median.json](configs/pruning/retinanet_coco_pruning_geometric_median.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median.tar.gz)|
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + filter pruning 40%|COCO 2017|32.67 (0.76)|[retinanet_coco_pruning_geometric_median_int8.json](configs/pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/retinanet_coco_pruning_geometric_median_int8.tar.gz)|


<a id="tensorflow_instance_segmentation"></a>
### Instance segmentation

|Model|Compression algorithm|Dataset|            mAP (_drop_) %             |NNCF config file|Checkpoint|
| :---: | :---: | :---: |:-------------------------------------:| :---: | :---: |
|Mask-R-CNN|None|COCO 2017|        bbox: 37.33 segm: 33.56        |[mask_rcnn_coco.json](configs/mask_rcnn_coco.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco.tar.gz)|
|Mask-R-CNN|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017| bbox: 37.19 (0.14) segm: 33.54 (0.02) |[mask_rcnn_coco_int8.json](configs/quantization/mask_rcnn_coco_int8.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_int8.tar.gz)|
|Mask-R-CNN|Magnitude sparsity, 50%|COCO 2017| bbox: 36.94 (0.39) segm: 33.23 (0.33) |[mask_rcnn_coco_magnitude_sparsity.json](configs/sparsity/mask_rcnn_coco_magnitude_sparsity.json)|[Download](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/tensorflow/mask_rcnn_coco_magnitude_sparsity.tar.gz)|


<a id="onnx_models"></a>
## ONNX models

<a id="onnx_classification"></a>
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


<a id="onnx_object_detection"></a>
### Object detection

|ONNX Model| Compression algorithm | Dataset |mAP (drop) %|
| :---: |:---------------------:| :---: | :---: |
|SSD1200| INT8 (Post-Training)  |COCO2017|20.17 (0.17)|
|Tiny-YOLOv2| INT8 (Post-Training)  |VOC12|29.03 (0.23)|
