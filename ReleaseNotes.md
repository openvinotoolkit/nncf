# Release Notes

## Introduction
*Neural Network Compression Framework (NNCF)* is a toolset for Neural Networks model compression.
The framework organized as a Python module that can be built and used as standalone or within
samples distributed with the code.  The samples demonstrate the usage of compression methods on
public models and datasets for three different use cases: Image Classification, Object Detection,
and Semantic Segmentation.

## New in Release 2.0:
- Added TensorFlow 2.4.2 support - NNCF can now be used to apply the compression algorithms to models originally trained in TensorFlow.
NNCF with TensorFlow backend supports the following features:
  - Compression algorithms:
    - Quantization (with HW-specific targeting aligned with PyTorch)
    - Sparsity:
      - Magnitude Sparsity 
      - RB Sparsity
    - Filter pruning
  - Support for only Keras models consisting of standard Keras layers and created by:
    - Keras Sequential API
    - Keras Functional API
  - Automatic, configurable model graph transformation to obtain the compressed model.
  - Distributed training on multiple GPUs on one machine is supported using `tf.distribute.MirroredStrategy`.
  - Exporting compressed models to SavedModel or Frozen Graph format, ready to use with OpenVINO™ toolkit.
  
- Added model compression samples for NNCF with TensorFlow backend:
  - Classification
    - Keras training loop.
    - Models form the tf.keras.applications module (ResNets, MobileNets, Inception and etc.) are supported.
    - TensorFlow Datasets (TFDS) and TFRecords (ImageNet2012, Cifar100, Cifar10) are supported.
    - Compression results are claimed for MobileNet V2, MobileNet V3 small, MobileNet V3 large, ResNet50, Inception V3.
  - Object Detection (Compression results are claimed for RetinaNet, YOLOv4)
    - Custom training loop.
    - TensorFlow Datasets (TFDS) and TFRecords for COCO2017 are supported.
    - Compression results for are claimed for RetinaNet, YOLOv4.
  - Instance Segmentation
    - Custom training loop
    - TFRecords for COCO2017 is supported.
    - Compression results are claimed for MaskRCNN

- Accuracy-aware training available for filter pruning and sparsity in order to achieve best compression results within a given accuracy drop threshold in a fully automated fashion.
- Framework-specific checkpoints produced with NNCF now have NNCF-specific compression state information included, so that the exact compressed model state can be restored/loaded without having to provide the same NNCF config file that was used during the creation of the NNCF-compressed checkpoint 
- Common interface for compression methods for both PyTorch and TensorFlow backends (https://github.com/openvinotoolkit/nncf/tree/develop/nncf/api).
- (PyTorch) Added an option to specify an effective learning rate multiplier for the trainable parameters of the compression algorithms via NNCF config, for finer control over which should tune faster - the underlying FP32 model weights or the compression parameters.
- (PyTorch) Unified scales for concat operations - the per-tensor quantizers that affect the concat operations will now have identical scales so that the resulting concatenated tensor can be represented without loss of accuracy w.r.t. the concatenated subcomponents.
- (TensorFlow) Algo-mixing: Added configuration files and reference checkpoints for filter-pruned + qunatized models: ResNet50@ImageNet2012(40% of filters pruned + INT8), RetinaNet@COCO2017(40% of filters pruned + INT8).
- (Experimental, PyTorch) [Learned Global Ranking]((https://arxiv.org/abs/1904.12368)) filter pruning mechanism for better pruning ratios with less accuracy drop for a broad range of models has been implemented.
- (Experimental, PyTorch) Knowledge distillation supported, ready to be used with any compression algorithm to produce an additional loss source of the compressed model against the uncompressed version

Breaking changes:
- `CompressionLevel` has been renamed to `CompressionStage`
- `"ignored_scopes"` and "target_scopes" no longer allow prefix matching - use full-fledged regular expression approach via {re} if anything more than an exact match is desired.
- (PyTorch) Removed version-agnostic name mapping for ReLU operations, i.e. the NNCF configs that referenced "RELU" (all caps) as an operation name will now have to reference an exact ReLU PyTorch function name such as "relu" or "relu_"
- (PyTorch) Removed the example of code modifications (Git patches and base commit IDs are provided) for [mmdetection](https://github.com/open-mmlab/mmdetection) repository.
- Batchnorm adaptation "forgetting" step has been removed since it has been observed to introduce accuracy degradation; the "num_bn_forget_steps" parameter in the corresponding NNCF config section has been removed.
- Framework-specific requirements no longer installed during `pip install nncf` or `python setup.py install` and are assumed to be present in the user's environment; the pip's "extras" syntax must be used to install the BKC requirements, e.g. by executing `pip install nncf[tf]`, `pip install nncf[torch]` or `pip install nncf[tf,torch]`
- `"quantizable_subgraph_patterns"` option removed from the NNCF config

Bugfixes:
- (PyTorch) Fixed a hang with batchnorm adaptation being applied in DDP mode
- (PyTorch) Fixed tracing of the operations that return NotImplemented

## New in Release 1.7.1:
Bugfixes:
- Fixed a bug with where compressed models that were supposed to return named tuples actually returned regular tuples
- Fixed an issue with batch norm adaptation-enabled compression runs hanging in the DDP scenario

## New in Release 1.7:
- Adjust Padding feature to support accurate execution of U4 on VPU - when setting "target_device" to "VPU", the training-time padding values for quantized convolutions will be adjusted to better reflect VPU inference process.
- Weighted layers that are "frozen" (i.e. have requires_grad set to False at compressed model creation time) are no longer considered for compression, to better handle transfer learning cases.
- Quantization algorithm now sets up quantizers without giving an option for requantization, which guarantees best performance, although at some cost to quantizer configuration flexibility.
- Pruning models with FCOS detection heads and instance normalization operations now supported
- Added a mean percentile initializer for the quantization algorithm
- Now possible to additionally quantize model outputs (separate control for each output quantization is supported)
- Models quantized for CPU now use effective 7-bit quantization for weights - the ONNX-exported model is still configured to use 8 bits for quantization, but only the middle 128 quanta of the total possible 256 are actually used, which allows for better OpenVINO inference accuracy alignment with PyTorch on non-VNNI CPUs
- Bumped target PyTorch version to 1.8.1 and relaxed package requirements constraints to allow installation into environments with PyTorch >=1.5.0

Notable bugfixes:
- Fixed bias pruning in depthwise convolution
- Made per-tensor quantization available for all operations that support per-channel quantization 
- Fixed progressive training performance degradation when an output tensor of an NNCF-compressed model is reused as its input.
- `pip install .` path of installing NNCF from a checked-out repository is now supported.
- Nested `with no_nncf_trace()` blocks now function as expected.
- NNCF compression API now formally abstract to guard against virtual function calls
- Now possible to load AutoQ and HAWQ-produced checkpoints to evaluate them or export to ONNX

Removed features:
- Pattern-based quantizer setup mode for quantization algorithm - due to its logic, it did not guarantee that all required operation inputs are ultimately quantized.


## New in Release 1.6:
- Added AutoQ - an AutoML-based mixed-precision initialization mode for quantization, which utilizes the power of reinforcement learning to select the best quantizer configuration for any model in terms of quality metric for a given HW architecture type.
- NNCF now supports inserting compression operations as pre-hooks to PyTorch operations, instead of abusing the post-hooking; the flexibility of quantization setups has been improved as a result of this change.
- Improved the pruning algorithm to group together dependent filters from different layers in the network and prune these together
- Extended the ONNX compressed model exporting interface with an option to explicitly name input and output tensors
- Changed the compression scheduler so that the correspondingepoch_step  and step methods should now be called in the beginning of the epoch and before the optimizer step (previously these were called in the end of the epoch and after the optimizer step respectively)
- Data-dependent compression algorithm initialization is now specified in terms of dataset samples instead of training batches, e.g. `"num_init_samples"` should be used in place of "num_init_steps" in NNCF config files.
- Custom user modules to be registered for compression can now be specified to be ignored for certain compression algorithms
- Batch norm adaptation now being applied by default for all compression algorithms
- Bumped target PyTorch version to 1.7.0
- Custom OpenVINO operations such as "FakeQuantize" that appear in NNCF-exported ONNX models now have their ONNX `domain` set to org.openvinotoolkit
- The quantization algorithm will now quantize nn.Embedding and nn.EmbeddingBag weights when targeting CPU
- Added an option to optimize logarithms of quantizer scales instead of scales themselves directly, a technique which improves convergence in certain cases
- Added reference checkpoints for filter-pruned models: UNet@Mapillary (25% of filters pruned), SSD300@VOC (40% of filters pruned)


## New in Release 1.5:
- Switched to using the propagation-based mode for quantizer setup by default. Compared to the previous default, pattern-based mode, the propagation-based mode better ensures that all the inputs to operations that can be quantized on a given type of hardware are quantized in accordance with what this hardware allows. Default target hardware is CPU - adjustable via `"target_device"` option in the NNCF config. More details can be found in [Quantization.md](./docs/compression_algorithms/Quantization.md#quantizer-setup-and-hardware-config-files).
- HAWQ mixed-precision initialization now supports a compression ratio parameter setting - set to 1 for a fully INT8 model, > 1 to increasingly allow lower bitwidth. The level of compression for each layer is defined by a product of the layer FLOPS and the quantization bitwidth.    
- HAWQ mixed-precision initialization allows specifying a more generic `criterion_fn` callable to calculate the related loss in case of complex output's post-processing or multiple losses.  
- Improved algorithm of assigning bitwidth for activation quantizers in HAWQ mixed-precision initialization. If after taking into account the corresponding rules of hardware config there're 
 multiple options for choosing bitwidth, it chooses a common bitwidth for all adjacent weight quantizers. Adjacent quantizers refer to all quantizers between inputs-quantizable layers.
- Custom user modules can be registered to have their `weight` attribute considered for compression using the @nncf.register_module
- Possible to perform quantizer linking in various points in graph - such quantizers will share the quantization parameters, trainable and non-trainable
- VPU HW config now uses unified scales for elementwise operations (utilising the quantizer linking mechanism)
- Range initialization configurations can now be specified on a per-layer basis
- Sparsity levels can now be applied separately for each layer
- Quantization "scope_overrides" config section now allows to set specific initializers and quantizer configuration
- Calculation of metrics representing the degree of quantization using the quantization algorithm - example scripts now display it if a quantization algorithm is used
- `create_compressed_model` now accepts a custom `wrap_inputs_fn` callable that should mark tensors among the model's `forward` arguments as "input" tensors for the model - useful for models that accept a list of tensors as their `forward` argument instead of tensors directly.
- `prepare_for_export` method added for `CompressionAlgorithmController` objects so that the users can signal the compressed model to finalize internal compression states and prepare for subsequent ONNX export
- GPT2 compression enabled, configuration file added to the `transformers` integration patch
- Added GoogLeNet as a filter-pruned sample model (with final checkpoints)

## New in Release 1.4:
- Models with filter pruning applied are now exportable to ONNX
- BatchNorm adaptation now available as a common compression algorithm initialization step - currently disabled by default, see `"batchnorm_adaptation"` config parameters in compression algorithm documentation (e.g. [Quantizer.md](docs/compression_algorithms/Quantization.md)) for instructions on how to enable it in NNCF config
- Major performance improvements for per-channel quantization training - now performs almost as fast as the per-tensor quantization training
- nn.Embedding and nn.Conv1d weights are now quantized by default
- Compression level querying is now available to determine current compression level (for purposes of choosing a correct "best" checkpoint during training)
- Generalized initializing data loaders to handle more interaction cases between a model and the associated data loader
- FP16 training supported for quantization
- Ignored scopes can now be set for the propagation-based quantization setup mode
- Per-optimizer stepping enabled as an option for polynomial sparsity scheduler
- Added an example config and model checkpoint for the ResNet50 INT8 + 50% sparsity (RB)

## New in Release 1.3.1
- Now using PyTorch 1.5 and CUDA 10.2 by default
- Support for exporting quantized models to ONNX checkpoints with standard ONNX v10 QuantizeLinear/DequantizeLinear pairs (8-bit quantization only)
- Compression algorithm initialization moved to the compressed model creation stage

## New in Release 1.3:
- Filter pruning algorithm added
- Mixed-precision quantization with manual and automatic (HAWQ-powered) precision setup
- Support for DistilBERT
- Selecting quantization parameters based on hardware configuration preset (CPU, GPU, VPU)
- Propagation-based quantizer position setup mode (quantizers are position as early in the network control flow graph as possible while keeping inputs of target operation quantized)
- Improved model graph tracing with introduction of input nodes and intermediate tensor shape tracking
- Updated third-party integration patches for consistency with NNCF release v1.3
- CPU-only installation mode for execution on machines without CUDA GPU hardware installed
- Docker images supplied for easier setup in container-based environments
- Usability improvements (NNCF config .JSON file validation by schema, less boilerplate code, separate logging and others)

## New in Release 1.2:
- Support for transformer-based networks quantization (tested on BERT and RoBERTa)
- Added instructions and Git patches for integrating NNCF into third-party repositories ([mmdetection](https://github.com/open-mmlab/mmdetection), [transformers](https://github.com/huggingface/transformers))
- Support for GNMT quantization
- Regular expression format support for specifying ignored/target scopes in config files - prefix the regex-enabled scope with {re}

## New in Release 1.1

- Binary networks using XNOR and DoReFa methods
- Asymmetric quantization scheme and per-channel quantization of Convolution
- 3D models support
- Support of integration into the [mmdetection](https://github.com/open-mmlab/mmdetection) repository
- Custom search patterns for FakeQuantize operation insertion
- Quantization of the model input by default
- Support of quantization of non-ReLU models (ELU, sigmoid, swish, hswish, and others)

## New in Release 1.0

- Support of symmetric quantization and two sparsity algorithms with fine-tuning
- Automatic model graph transformation. The model is wrapped by the custom class and additional layers are inserted in the graph. The transformations are configurable.
- Three training samples which demonstrate usage of compression methods from the NNCF:
    - Image Classification:  torchvision models for classification and custom models on ImageNet and CIFAR10/100 datasets.
    - Object Detection: SSD300, SSD512, MobileNet SSD on Pascal VOC2007, Pascal VOC2012, and COCO datasets.
    - Semantic Segmentation: UNet, ICNet on CamVid and Mapillary Vistas datasets.
- Unified interface for compression methods.
- GPU-accelerated *Quantization* layer for fast model fine-tuning.
- Distributed training support in all samples.
- Configuration file examples for sparsity, quantization and sparsity with quantization for all three samples. Each type of compression requires only one additional stage of fine-tuning.
- Export models to the ONNX format that is supported by the [OpenVINO](https://github.com/opencv/dldt) toolkit.
