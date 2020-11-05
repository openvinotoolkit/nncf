# Release Notes

## Introduction
*Neural Network Compression Framework (NNCF)* is a toolset for Neural Networks model compression.
The framework organized as a Python module that can be built and used as standalone or within
samples distributed with the code.  The samples demonstrate the usage of compression methods on
public models and datasets for three different use cases: Image Classification, Object Detection,
and Semantic Segmentation.

## New in Release 1.5:
- Switched to using the propagation-based mode for quantizer setup by default. Compared to the previous default, pattern-based mode, the propagation-based mode better ensures that all the inputs to operations that can be quantized on a given type of hardware are quantized in accordance with what this hardware allows. Default target hardware is CPU - adjustable via `"target_device"` option in the NNCF config.
- HAWQ mixed precision initialization now supports a compression ration parameter setting - set to 1 for a fully INT8 model, > 1 to increasingly allow lower bit widths.
- HAWQ mixed precision initialization allows to specify a more generic `criterion_fn` callable to calculate the related loss in case of complex loss input argument signatures
- Custom user modules can be registered to have their `weight` attribute considered for compression using the @nncf.register_module
- Possible to perform quantizer linking in various points in graph - such quantizers will share the quantization parameters, trainable and non-trainable
- VPU HW config now uses unified scales for elementwise operations (utilising the quantizer linking mechanism)
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
