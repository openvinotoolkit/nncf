# Release Notes

## New in Release 2.14.1

Post-training Quantization:

- Bugfixes:
  - (PyTorch) Fixed the `get_torch_compile_wrapper` function to match with the `torch.compile`.
  - (OpenVINO) Updated cache statistics functionality to utilize the `safetensors` approach.

## New in Release 2.14.0

Post-training Quantization:

- Features:
  - Introduced `backup_mode` optional parameter in `nncf.compress_weights()` to specify the data type for embeddings, convolutions and last linear layers during 4-bit weights compression. Available options are INT8_ASYM by default, INT8_SYM, and NONE which retains the original floating-point precision of the model weights.
  - Added the `quantizer_propagation_rule` parameter, providing fine-grained control over quantizer propagation. This advanced option is designed to improve accuracy for models where quantizers with different granularity could be merged to per-tensor, potentially affecting model accuracy.
  - Introduced `nncf.data.generate_text_data` API method that utilizes LLM to generate data for further data-aware optimization. See the [example](examples/llm_compression/openvino/tiny_llama_synthetic_data/) for details.
  - (OpenVINO) Extended support of data-free and data-aware weight compression methods for `nncf.compress_weights()` with NF4 per-channel quantization, which makes compressed LLMs more accurate and faster on NPU.
  - (OpenVINO) Introduced a new option `statistics_path` to cache and reuse statistics for `nncf.compress_weights()`, reducing the time required to find optimal compression configurations. See the [TinyLlama example](examples/llm_compression/openvino/tiny_llama_find_hyperparams) for details.
  - (TorchFX, Experimental) Added support for quantization and weight compression of [Torch FX](https://pytorch.org/docs/stable/fx.html) models. The compressed models can be directly executed via `torch.compile(compressed_model, backend="openvino")` (see details [here](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html)). Added [INT8 quantization example](examples/post_training_quantization/torch_fx/resnet18). The list of supported features:
    - INT8 quantization with SmoothQuant, MinMax, FastBiasCorrection, and BiasCorrection algorithms via `nncf.quantize()`.
    - Data-free INT8, INT4, and mixed-precision weights compression with `nncf.compress_weights()`.
  - (PyTorch, Experimental) Added model tracing and execution pre-post hooks based on TorchFunctionMode.
- Fixes:
  - Resolved an issue with redundant quantizer insertion before elementwise operations, reducing noise introduced by quantization.
  - Fixed type mismatch issue for `nncf.quantize_with_accuracy_control()`.
  - Fixed BiasCorrection algorithm for specific branching cases.
  - (OpenVINO) Fixed GPTQ weight compression method for Stable Diffusion models.
  - (OpenVINO) Fixed issue with the variational statistics processing for `nncf.compress_weights()`.
  - (PyTorch, ONNX) Scaled dot product attention pattern quantization setup is aligned with OpenVINO.
- Improvements:
  - Reduction in peak memory by 30-50% for data-aware `nncf.compress_weights()` with AWQ, Scale Estimation, LoRA and mixed-precision algorithms.
  - Reduction in compression time by 10-20% for `nncf.compress_weights()` with AWQ algorithm.
  - Aligned behavior for ignored subgraph between different `networkx` versions.
  - Extended ignored patterns with RoPE block for `nncf.ModelType.TRANSFORMER` scheme.
  - (OpenVINO) Extended to the ignored scope for `nncf.ModelType.TRANSFORMER` scheme with GroupNorm metatype.
  - (ONNX) SE-block ignored pattern variant for `torchvision` mobilenet_v3 has been extended.
- Tutorials:
  - [Post-Training Optimization of Llama-3.2-11B-Vision Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/mllama-3.2/mllama-3.2.ipynb)
  - [Post-Training Optimization of YOLOv11 Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov11-optimization/yolov11-object-detection.ipynb)
  - [Post-Training Optimization of Whisper in Automatic speech recognition with OpenVINO Generate API](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/whisper-asr-genai/whisper-asr-genai.ipynb)
  - [Post-Training Optimization of Pixtral Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pixtral/pixtral.ipynb)
  - [Post-Training Optimization of LLM ReAct Agent Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-agent-react/llm-agent-react.ipynb)
  - [Post-Training Optimization of CatVTON Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/catvton/catvton.ipynb)
  - [Post-Training Optimization of Stable Diffusion v3 Model in Torch FX Representation](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/stable-diffusion-v3/stable-diffusion-v3-torch-fx.ipynb)
- Known issues:
  - (ONNX) `nncf.quantize()` method can generate inaccurate INT8 results for MobileNet models with the BiasCorrection algorithm.  

Deprecations/Removals:

- Migrated from using `setup.py` to `pyproject.toml` for the build and package configuration. It is aligned with Python packaging standards as outlined in PEP 517 and PEP 518. The installation through `setup.py` does not work anymore. No impact on the installation from PyPI and Conda.
- Removed support for Python 3.8.
- (PyTorch) `nncf.torch.create_compressed_model()` function has been deprecated.

Requirements:

- Updated ONNX (1.17.0) and ONNXRuntime (1.19.2) versions.
- Updated PyTorch (2.5.1) and Torchvision (0.20.1) versions.
- Updated NumPy (<2.2.0) version support.
- Updated Ultralytics (8.3.22) version.

## New in Release 2.13.0

Post-training Quantization:

- Features:`
  - (OpenVINO) Added support for combining GPTQ with AWQ and Scale Estimation (SE) algorithms in `nncf.compress_weights()` for more accurate weight compression of LLMs. Thus, the following combinations with GPTQ are now supported: AWQ+GPTQ+SE, AWQ+GPTQ, GPTQ+SE, GPTQ.
  - (OpenVINO) Added LoRA Correction Algorithm to further improve the accuracy of int4 compressed models on top of other algorithms - AWQ and Scale Estimation. It can be enabled via the optional `lora_correction` parameter of the `nncf.compress_weights()` API. The algorithm increases compression time and incurs a negligible model size overhead. Refer to [accuracy/footprint trade-off](docs/usage/post_training_compression/weights_compression/Usage.md#accuracyfootprint-trade-off) for different int4 compression methods.
  - (PyTorch) Added implementation of the experimental Post-training Activation Pruning algorithm. Refer to [Activation Sparsity](nncf/experimental/torch/sparsify_activations/ActivationSparsity.md) for details.
  - Added a memory monitoring tool for logging the memory a piece of python code or a script allocates. Refer to [NNCF tools](tools/README.md) for details.
- Fixes:
  - (OpenVINO) Fixed the quantization of Convolution and LSTMSequence operations in cases where some inputs are part of a ShapeOF subgraph.
  - (OpenVINO) Fixed issue with the FakeConvert duplication for FP8.
  - Fixed Smooth Quant algorithm issue in case of the incorrect shapes.
  - Fixed non-deterministic layer-wise scheduling.
- Improvements:
  - (OpenVINO) Increased hardware-fused pattern coverage.
  - Improved progress bar logic during weights compression for more accurate remaining time estimation.
  - Extended Scale estimation bitness range support for the `nncf.compress_weights()`.
  - Removed extra logging for the algorithm-generated ignored scope.
- Tutorials:
  - [Post-Training Optimization of Flux.1 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/flux.1-image-generation/flux.1-image-generation.ipynb)
  - [Post-Training Optimization of PixArt-α Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pixart/pixart.ipynb)
  - [Post-Training Optimization of InternVL2 Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/internvl2/internvl2.ipynb)
  - [Post-Training Optimization of Qwen2Audio Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen2-audio/qwen2-audio.ipynb)
  - [Post-Training Optimization of NuExtract Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/nuextract-structure-extraction/nuextract-structure-extraction.ipynb)
  - [Post-Training Optimization of MiniCPM-V2 Model](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/minicpm-v-multimodal-chatbot/minicpm-v-multimodal-chatbot.ipynb)

Compression-aware training:

- Fixes:
  - (PyTorch) Fixed some scenarios of NNCF patching interfering with `torch.compile`.

Requirements:

- Updated PyTorch (2.4.0) and Torchvision (0.19.0) versions.

## New in Release 2.12.0

Post-training Quantization:

- Features:
  - (OpenVINO, PyTorch, ONNX) Excluded comparison operators from the quantization scope for `nncf.ModelType.TRANSFORMER`.
  - (OpenVINO, PyTorch) Changed the representation of symmetrically quantized weights from an unsigned integer with a fixed zero-point to a signed data type without a zero-point in the `nncf.compress_weights()` method.
  - (OpenVINO) Extended patterns support of the AWQ algorithm as part of `nncf.compress_weights()`. This allows apply AWQ for the wider scope of the models.
  - (OpenVINO) Introduced `nncf.CompressWeightsMode.E2M1`  `mode` option of `nncf.compress_weights()` as the new MXFP4 precision (Experimental).
  - (OpenVINO) Added support for models with BF16 precision in the `nncf.quantize()` method.
  - (PyTorch) Added quantization support for the `torch.addmm`.
  - (PyTorch) Added quantization support for the `torch.nn.functional.scaled_dot_product_attention`.
- Fixes:
  - (OpenVINO, PyTorch, ONNX) Fixed Fast-/BiasCorrection algorithms with correct support of transposed MatMul layers.
  - (OpenVINO) Fixed `nncf.IgnoredScope()` functionality for models with If operation.
  - (OpenVINO) Fixed patterns with PReLU operations.
  - Fixed runtime error while importing NNCF without Matplotlib package.
- Improvements:
  - Reduced the amount of memory required for applying `nncf.compress_weights()` to OpenVINO models.
  - Improved logging in case of the not empty `nncf.IgnoredScope()`.
- Tutorials:
  - [Post-Training Optimization of Stable Audio Open Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-audio/stable-audio.ipynb)
  - [Post-Training Optimization of Phi3-Vision Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/phi-3-vision/phi-3-vision.ipynb)
  - [Post-Training Optimization of MiniCPM-V2 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/minicpm-v-multimodal-chatbot/minicpm-v-multimodal-chatbot.ipynb)
  - [Post-Training Optimization of Jina CLIP Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/jina-clip/jina-clip.ipynb)
  - [Post-Training Optimization of Stable Diffusion v3 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-diffusion-v3/stable-diffusion-v3.ipynb)
  - [Post-Training Optimization of HunyuanDIT Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hunyuan-dit-image-generation/hunyuan-dit-image-generation.ipynb)
  - [Post-Training Optimization of DDColor Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/ddcolor-image-colorization/ddcolor-image-colorization.ipynb)
  - [Post-Training Optimization of DynamiCrafter Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/dynamicrafter-animating-images/dynamicrafter-animating-images.ipynb)
  - [Post-Training Optimization of DepthAnythingV2 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/depth-anything/depth-anything-v2.ipynb)
  - [Post-Training Optimization of Kosmos-2 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/kosmos2-multimodal-large-language-model/kosmos2-multimodal-large-language-model.ipynb)

Compression-aware training:

- Fixes:
  - (PyTorch) Fixed issue with wrapping for operator without patched state.

Requirements:

- Updated Tensorflow (2.15) version. This version requires Python 3.9-3.11.

## New in Release 2.11.0

Post-training Quantization:

- Features:
  - (OpenVINO) Added Scale Estimation algorithm for 4-bit data-aware weights compression. The optional `scale_estimation` parameter was introduced to `nncf.compress_weights()` and can be used to minimize accuracy degradation of compressed models (note that this algorithm increases the compression time).
  - (OpenVINO) Added GPTQ algorithm for 8/4-bit data-aware weights compression, supporting INT8, INT4, and NF4 data types. The optional `gptq` parameter was introduced to `nncf.compress_weights()` to enable the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm.
  - (OpenVINO) Added support for models with BF16 weights in the weights compression method, `nncf.compress_weights()`.
  - (PyTorch) Added support for quantization and weight compression of the custom modules.
- Fixes:
  - (OpenVINO) Fixed incorrect node with bias determination in Fast-/BiasCorrection and ChannelAlighnment algorithms.
  - (OpenVINO, PyTorch) Fixed incorrect behaviour of `nncf.compress_weights()` in case of compressed model as input.
  - (OpenVINO, PyTorch) Fixed SmoothQuant algorithm to work with Split ports correctly.
- Improvements:
  - (OpenVINO) Aligned resulting compression subgraphs for the `nncf.compress_weights()` in different FP precisions.
  - Aligned 8-bit scheme for NPU target device with the CPU.
- Examples:
  - (OpenVINO, ONNX) Updated ignored scope for YOLOv8 examples utilizing a subgraphs approach.
- Tutorials:
  - [Post-Training Optimization of Stable Video Diffusion Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-video-diffusion/stable-video-diffusion.ipynb)
  - [Post-Training Optimization of YOLOv10 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov10-optimization/yolov10-optimization.ipynb)
  - [Post-Training Optimization of LLaVA Next Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/nano-llava-multimodal-chatbot/nano-llava-multimodal-chatbot.ipynb)
  - [Post-Training Optimization of S3D MIL-NCE Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/s3d-mil-nce-text-to-video-retrieval/s3d-mil-nce-text-to-video-retrieval.ipynb)
  - [Post-Training Optimization of Stable Cascade Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-cascade-image-generation/stable-cascade-image-generation.ipynb)

Compression-aware training:

- Features:
  - (PyTorch) `nncf.quantize` method is now the recommended path for the quantization initialization for Quantization-Aware Training.
  - (PyTorch) Compression modules placement in the model now can be serialized and restored with new API functions: `compressed_model.nncf.get_config()` and `nncf.torch.load_from_config`. The [documentation](/docs/usage/training_time_compression/quantization_aware_training/Usage.md#saving-and-loading-compressed-models) for the saving/loading of a quantized model is available, and Resnet18 [example](examples/quantization_aware_training/torch/resnet18) was updated to use the new API.
- Fixes:
  - (PyTorch) Fixed compatibility with `torch.compile`.
- Improvements:
  - (PyTorch) Base parameters were extended for the EvolutionOptimizer (LeGR algorithm part).
  - (PyTorch) Improved wrapping for parameters which are not tensors.
- Examples:
  - (PyTorch) Added [an example](examples/quantization_aware_training/torch/anomalib) for STFPM model from Anomalib.
- Tutorials:
  - [Quantization-Sparsity Aware Training of PyTorch ResNet-50 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-sparsity-aware-training/pytorch-quantization-sparsity-aware-training.ipynb)

Deprecations/Removals:

- Removed extra dependencies to install backends from setup.py (like `[torch]` are `[tf]`, `[onnx]` and `[openvino]`).
- Removed `openvino-dev` dependency.

Requirements:

- Updated PyTorch (2.2.1) and Torchvision (0.18.0) versions.

## New in Release 2.10.0

Post-training Quantization:

- Features:
  - Introduced the subgraph defining functionality for the `nncf.IgnoredScope()` option.
  - Introduced limited support for the batch size of more than 1. MobilenetV2 [PyTorch example](examples/post_training_quantization/torch/mobilenet_v2) was updated with batch support.
- Fixes:
  - Fixed issue with the `nncf.OverflowFix` parameter absence in some scenarios.
  - Aligned the list of correctable layers for the FastBiasCorrection algorithm between PyTorch, OpenVINO and ONNX backends.
  - Fixed issue with the `nncf.QuantizationMode` parameters combination.
  - Fixed MobilenetV2 ([PyTorch](examples/post_training_quantization/torch/mobilenet_v2), [ONNX](examples/post_training_quantization/onnx/mobilenet_v2), [OpenVINO](examples/post_training_quantization/openvino/mobilenet_v2)) examples for the Windows platform.
  - (OpenVINO) Fixed [Anomaly Classification example](examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control) for the Windows platform.
  - (PyTorch) Fixed bias shift magnitude calculation for fused layers.
  - (OpenVINO) Fixed removing the ShapeOf graph which led to an error in the `nncf.quantize_with_accuracy_control()` method.
- Improvements:
  - `OverflowFix`, `AdvancedSmoothQuantParameters` and `AdvancedBiasCorrectionParameters` were exposed into the `nncf.*` namespace.
  - (OpenVINO, PyTorch) Introduced scale compression to FP16 for weights in `nncf.compress_weights()` method, regardless of model weights precision.
  - (PyTorch) Modules that NNCF inserted were excluded from parameter tracing.
  - (OpenVINO) Extended the list of correctable layers for the BiasCorrection algorithm.
  - (ONNX) Aligned BiasCorrection algorithm behaviour with OpenVINO in specific cases.
- Tutorials:
  - [Post-Training Optimization of PhotoMaker Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/photo-maker/photo-maker.ipynb)
  - [Post-Training Optimization of Stable Diffusion XL Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-diffusion-xl/stable-diffusion-xl.ipynb)
  - [Post-Training Optimization of KerasCV Stable Diffusion Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/stable-diffusion-keras-cv/stable-diffusion-keras-cv.ipynb)
  - [Post-Training Optimization of Paint By Example Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paint-by-example/paint-by-example.ipynb)
  - [Post-Training Optimization of aMUSEd Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/amused-lightweight-text-to-image/amused-lightweight-text-to-image.ipynb)
  - [Post-Training Optimization of InstantID Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/instant-id/instant-id.ipynb)
  - [Post-Training Optimization of LLaVA Next Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llava-next-multimodal-chatbot/llava-next-multimodal-chatbot.ipynb)
  - [Post-Training Optimization of AnimateAnyone Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/animate-anyone/animate-anyone.ipynb)
  - [Post-Training Optimization of YOLOv8-OBB Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization/yolov8-obb.ipynb)
  - [Post-Training Optimization of LLM Agent](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-agent-langchain/llm-agent-langchain.ipynb)

Compression-aware training:

- Features:
  - (PyTorch) `nncf.quantize` method now may be used as quantization initialization for Quantization-Aware Training. Added a [Resnet18-based example](examples/quantization_aware_training/torch/resnet18) with the transition from the Post-Training Quantization to a Quantization-Aware Training algorithm.
  - (PyTorch) Introduced extractors for the fused Convolution, Batch-/GroupNorm, and Linear functions.
- Fixes:
  - (PyTorch) Fixed `apply_args_defaults` function issue.
  - (PyTorch) Fixed `dtype` handling for the compressed `torch.nn.Parameter`.
  - (PyTorch) Fixed `is_shared` parameter propagation.
- Improvements:
  - (PyTorch) Updated command creation behaviour to reduce the number of adapters.
  - (PyTorch) Added option to insert point for models that wrapped with `replace_modules=False`.
- Deprecations/Removals:
  - (PyTorch) Removed the `binarization` algorithm.
  - NNCF installation via `pip install nncf[<framework>]` option is now deprecated.
- Requirements:
  - Updated PyTorch (2.2.1) and CUDA (12.1) versions.
  - Updated ONNX (1.16.0) and ONNXRuntime (1.17.1) versions.

## New in Release 2.9.0

Post-training Quantization:

- Features:
  - (OpenVINO) Added modified AWQ algorithm for 4-bit data-aware weights compression. This algorithm applied only for patterns `MatMul->Multiply->Matmul`. For that `awq` optional parameter has been added to `nncf.compress_weights()` and can be used to minimize accuracy degradation of compressed models (note that this option increases the compression time).
  - (ONNX) Introduced support for the ONNX backend in the `nncf.quantize_with_accuracy_control()` method. Users can now perform quantization with accuracy control for `onnx.ModelProto`. By leveraging this feature, users can enhance the accuracy of quantized models while minimizing performance impact.
  - (ONNX) Added an example based on the YOLOv8n-seg model for demonstrating the usage of quantization with accuracy control for the ONNX backend.
  - (PT) Added SmoothQuant algorithm for PyTorch backend in `nncf.quantize()`.
  - (OpenVINO) Added [an example](examples/llm_compression/openvino/tiny_llama_find_hyperparams) with the hyperparameters tuning for the TinyLLama model.
  - Introduced the `nncf.AdvancedAccuracyRestorerParameters`.
  - Introduced the `subset_size` option for the `nncf.compress_weights()`.
  - Introduced `TargetDevice.NPU` as the replacement for `TargetDevice.VPU`.
- Fixes:
  - Fixed API Enums serialization/deserialization issue.
  - Fixed issue with required arguments for `revert_operations_to_floating_point_precision` method.
- Improvements:
  - (ONNX) Aligned statistics collection with OpenVINO and PyTorch backends.
  - Extended `nncf.compress_weights()` with Convolution & Embeddings compression in order to reduce memory footprint.
- Deprecations/Removals:
  - (OpenVINO) Removed outdated examples with `nncf.quantize()` for BERT and YOLOv5 models.
  - (OpenVINO) Removed outdated example with `nncf.quantize_with_accuracy_control()` for SSD MobileNetV1 FPN model.
  - (PyTorch) Deprecated the `binarization` algorithm.
  - Removed Post-training Optimization Tool as OpenVINO backend.
  - Removed Dockerfiles.
  - `TargetDevice.VPU` was replaced by `TargetDevice.NPU`.
- Tutorials:
  - [Post-Training Optimization of Stable Diffusion v2 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/236-stable-diffusion-v2/236-stable-diffusion-v2-text-to-image.ipynb)
  - [Post-Training Optimization of DeciDiffusion Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/259-decidiffusion-image-generation/259-decidiffusion-image-generation.ipynb)
  - [Post-Training Optimization of DepthAnything Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/280-depth-anything/280-depth-anything.ipynb)
  - [Post-Training Optimization of Stable Diffusion ControlNet Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/235-controlnet-stable-diffusion/235-controlnet-stable-diffusion.ipynb)

Compression-aware training:

- Fixes
  - (PyTorch) Fixed issue with `NNCFNetworkInterface.get_clean_shallow_copy` missed arguments.

## New in Release 2.8.1

Post-training Quantization:

- Bugfixes:
  - (Common) Fixed issue with `nncf.compress_weights()` to avoid overflows on 32-bit Windows systems.
  - (Common) Fixed performance issue with `nncf.compress_weights()` on LLama models.
  - (Common) Fixed `nncf.quantize_with_accuracy_control` pipeline with `tune_hyperparams=True` enabled option.
  - (OpenVINO) Fixed issue for stateful LLM models and added state restoring after the inference for it.
  - (PyTorch) Fixed issue with `nncf.compress_weights()` for LLM models with the executing `is_floating_point` with tracing.

## New in Release 2.8.0

Post-training Quantization:

- Breaking changes:
  - `nncf.quantize` signature has been changed to add `mode: Optional[nncf.QuantizationMode] = None` as its 3-rd argument, between the original `calibration_dataset` and `preset` arguments.
  - (Common) `nncf.common.quantization.structs.QuantizationMode` has been renamed to `nncf.common.quantization.structs.QuantizationScheme`
- General:
  - (OpenVINO) Changed default OpenVINO opset from 9 to 13.
- Features:
  - (OpenVINO) Added 4-bit data-aware weights compression. For that `dataset` optional parameter has been added to `nncf.compress_weights()` and can be used to minimize accuracy degradation of compressed models (note that this option increases the compression time).
  - (PyTorch) Added support for PyTorch models with shared weights and custom PyTorch modules in nncf.compress_weights(). The weights compression algorithm for PyTorch models is now based on tracing the model graph. The dataset parameter is now required in nncf.compress_weights() for the compression of PyTorch models.
  - (Common) Renamed the `nncf.CompressWeightsMode.INT8` to `nncf.CompressWeightsMode.INT8_ASYM` and introduce `nncf.CompressWeightsMode.INT8_SYM` that can be efficiently used with dynamic 8-bit quantization of activations.
  The original `nncf.CompressWeightsMode.INT8` enum value is now deprecated.
  - (OpenVINO) Added support for quantizing the ScaledDotProductAttention operation from OpenVINO opset 13.
  - (OpenVINO) Added FP8 quantization support via `nncf.QuantizationMode.FP8_E4M3` and `nncf.QuantizationMode.FP8_E5M2` enum values, invoked via passing one of these values as an optional `mode` argument to `nncf.quantize`. Currently, OpenVINO supports inference of FP8-quantized models in reference mode with no performance benefits and can be used for accuracy projections.
  - (Common) Post-training Quantization with Accuracy Control - `nncf.quantize_with_accuracy_control()` has been extended by `restore_mode` optional parameter to revert weights to int8 instead of the original precision.
  This parameter helps to reduce the size of the quantized model and improves its performance.
  By default, it's disabled and model weights are reverted to the original precision in `nncf.quantize_with_accuracy_control()`.
  - (Common) Added an `all_layers: Optional[bool] = None` argument to `nncf.compress_weights` to indicate whether embeddings and last layers of the model should be compressed to a primary precision. This is relevant to 4-bit quantization only.
  - (Common) Added a `sensitivity_metric: Optional[nncf.parameters.SensitivityMetric] = None` argument to `nncf.compress_weights` for finer control over the sensitivity metric for assigning quantization precision to layers.
  Defaults to weight quantization error if a dataset is not provided for weight compression and to maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise if a dataset is provided.
  By default, the backup precision is assigned for the embeddings and last layers.
- Fixes:
  - (OpenVINO) Models with embeddings (e.g. `gpt-2`, `stable-diffusion-v1-5`, `stable-diffusion-v2-1`, `opt-6.7b`, `falcon-7b`, `bloomz-7b1`) are now more accurately quantized.
  - (PyTorch) `nncf.strip(..., do_copy=True)` now actually returns a deepcopy (stripped) of the model object.
  - (PyTorch) Post-hooks can now be set up on operations that return `torch.return_type` (such as `torch.max`).
  - (PyTorch) Improved dynamic graph tracing for various tensor operations from `torch` namespace.
  - (PyTorch) More robust handling of models with disjoint traced graphs when applying PTQ.
- Improvements:
  - Reformatted the tutorials section in the top-level `README.md` for better readability.
- Deprecations/Removals:
  - (Common) The original `nncf.CompressWeightsMode.INT8` enum value is now deprecated.
  - (PyTorch) The Git patch for integration with HuggingFace `transformers` repository is marked as deprecated and will be removed in a future release.
  Developers are advised to use [optimum-intel](https://github.com/huggingface/optimum-intel) instead.
  - Dockerfiles in the NNCF Git repository are deprecated and will be removed in a future release.

## New in Release 2.7.0

Post-training Quantization:

- Features:
  - (OpenVINO) Added support for data-free 4-bit weights compression through NF4 and INT4 data types (`compress_weights(…)` pipeline).
  - (OpenVINO) Added support for [IF operation](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_If_8.html) quantization.
  - (OpenVINO) Added `dump_intermediate_model` parameter support for AccuracyAwareAlgorithm (`quantize_with_accuracy_control(…)` pipeline).
  - (OpenVINO) Added support for SmoothQuant and ChannelAlignment algorithms for HyperparameterTuner algorithm (`quantize_with_tune_hyperparams(…)` pipeline).
  - (PyTorch) Post-training Quantization is now supported with `quantize(…)` pipeline and the common implementation of quantization algorithms. Deprecated `create_compressed_model()` method for Post-training Quantization.
  - Added new types (AvgPool, GroupNorm, LayerNorm) to the ignored scope for `ModelType.Transformer` scheme.
  - `QuantizationPreset.Mixed` was set as the default for `ModelType.Transformer` scheme.
- Fixes:
  - (OpenVINO, ONNX, PyTorch) Aligned/added patterns between backends (SE block, MVN layer, multiple activations, etc.) to restore performance/metrics.
  - Fixed patterns for `ModelType.Transformer` to align with the [quantization scheme](https://docs.openvino.ai/latest/openvino_docs_OV_UG_lpt.html).
- Improvements:
  - Improved UX with the new progress bar for pipeline, new exceptions, and .dot graph visualization updates.
  - (OpenVINO) Optimized WeightsCompression algorithm (`compress_weights(…)` pipeline) execution time for LLM's quantization, added ignored scope support.
  - (OpenVINO) Optimized AccuracyAwareQuantization algorithm execution time with multi-threaded approach while calculating ranking score (`quantize_with_accuracy_control(…)` pipeline).
  - (OpenVINO) Added [extract_ov_subgraph tool](tools/extract_ov_subgraph.py) for large IR subgraph extraction.
  - (ONNX) Optimized quantization pipeline (up to 1.15x speed up).
- Tutorials:
  - [Post-Training Optimization of BLIP Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/233-blip-visual-language-processing)
  - [Post-Training Optimization of DeepFloyd IF Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/238-deepfloyd-if)
  - [Post-Training Optimization of Grammatical Error Correction Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/214-grammar-correction)
  - [Post-Training Optimization of Dolly 2.0 Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/240-dolly-2-instruction-following)
  - [Post-Training Optimization of Massively Multilingual Speech Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/255-mms-massively-multilingual-speech)
  - [Post-Training Optimization of OneFormer Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/249-oneformer-segmentation)
  - [Post-Training Optimization of InstructPix2Pix Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/231-instruct-pix2pix-image-editing)
  - [Post-Training Optimization of LLaVA Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/257-llava-multimodal-chatbot)
  - [Post-Training Optimization of Latent Consistency Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/263-latent-consistency-models-image-generation)
  - [Post-Training Optimization of Distil-Whisper Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/267-distil-whisper-asr)
  - [Post-Training Optimization of FastSAM Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/261-fast-segment-anything)
- Known issues:
  - (ONNX) `quantize(...)` method can generate inaccurate int8 results for models with the BatchNormalization layer that contains biases. To get the best accuracy, use the `do_constant_folding=True` option during export from PyTorch to ONNX.

Compression-aware training:

- Fixes:
  - (PyTorch) Fixed Hessian trace calculation to solve [#2155](https://github.com/openvinotoolkit/nncf/issues/2155) issue.
- Requirements:
  - Updated PyTorch version (2.1.0).
  - Updated numpy version (<1.27).
- Deprecations/Removals:
  - (PyTorch) Removed legacy external quantizer storage names.
  - (PyTorch) Removed torch < 2.0 version support.

## New in Release 2.6.0

Post-training Quantization:

- Features:
  - Added `CPU_SPR` device type support.
  - Added quantizers scales unification.
  - Added quantization scheme for ReduceSum operation.
  - Added new types (ReduceL2, ReduceSum, Maximum) to the ignored scope for `ModelType.Transformer`.
  - (OpenVINO) Added SmoothQuant algorithm.
  - (OpenVINO) Added ChannelAlignment algorithm.
  - (OpenVINO) Added HyperparameterTuner algorithm.
  - (PyTorch) Added FastBiasCorrection algorithm support.
  - (OpenVINO, ONNX) Added embedding weights quantization.
  - (OpenVINO, PyTorch) Added new `compress_weights` method that provides data-free [INT8 weights compression](docs/usage/post_training_compression/weights_compression/Usage.md).
- Fixes:
  - Fixed detection of decomposed post-processing in models.
  - Multiple fixes (new patterns, bugfixes, etc.) to solve [#1936](https://github.com/openvinotoolkit/nncf/issues/1936) issue.
  - Fixed model reshaping while quantization to keep original model shape.
  - (OpenVINO) Added support for sequential models quanitzation.
  - (OpenVINO) Fixed in-place statistics cast to support empty dimensions.
  - (OpenVINO, ONNX) Fixed quantization of the MatMul operation with weights rank > 2.
  - (OpenVINO, ONNX) Fixed BiasCorrection algorithm to enable [CLIP model quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/228-clip-zero-shot-image-classification).
- Improvements:
  - Optimized `quantize(…)` pipeline (up to 4.3x speed up in total).
  - Optimized `quantize_with_accuracy_control(…)` pipelilne (up to 8x speed up for [122-quantizing-model-with-accuracy-control](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/122-quantizing-model-with-accuracy-control) notebook).
  - Optimized general statistics collection (up to 1.2x speed up for ONNX backend).
  - Ignored patterns separated from Fused patterns scheme (with multiple patterns addition).
- Tutorials:
  - [Post-Training Optimization of Segment Anything Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/237-segment-anything).
  - [Post-Training Optimization of CLIP Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/228-clip-zero-shot-image-classification).
  - [Post-Training Optimization of ImageBind Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/239-image-bind).
  - [Post-Training Optimization of Whisper Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/227-whisper-subtitles-generation).
  - [Post-Training Optimization with accuracy control](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/122-quantizing-model-with-accuracy-control).

Compression-aware training:

- Features:
  - Added shape pruning processor for BootstrapNAS algorithm.
  - Added KD loss for BootstrapNAS algorithm.
  - Added `validate_scopes` parameter for NNCF configuration.
  - (PyTorch) Added PyTorch 2.0 support.
  - (PyTorch) Added `.strip()` option to API.
  - (PyTorch) Enabled bfloat data type for quantization kernels.
  - (PyTorch) Quantized models can now be `torch.jit.trace`d without calling `.strip()`.
  - (PyTorch) Added support for overridden `forward` instance attribute on model objects passed into `create_compressed_model`.
  - (Tensorflow) Added Tensorflow 2.12 support.
- Fixes:
  - (PyTorch) Fixed padding adjustment issue in the elastic kernel to work with the different active kernel sizes.
  - (PyTorch) Fixed the torch graph tracing in the case the tensors belonging to parallel edges are interleaved in the order of the tensor argument.
  - (PyTorch) Fixed recurrent nodes matching (LSTM, GRU cells) condition with the strict rule to avoid adding not necessary nodes to the ignored scope.
  - (PyTorch) Fixed `torch.jit.script` wrapper so that user-side handling exceptions during `torch.jit.script` invocation do not cause NNCF to be permanently disabled.
  - (PyTorch, Tensorflow) Adjusted quantizer propagation algorithm to check if quantizer propagation will result in output quantization.
  - (PyTorch) Added redefined `__class__` method for ProxyModule that avoids causing error while calling `.super()` in forward method.
- Deprecations/Removals:
  - (PyTorch) Removed deprecated `NNCFNetwork.__getattr__`, `NNCFNetwork.get_nncf_wrapped_model` methods.
- Requirements:
  - Updated PyTorch version (2.0.1).
  - Updated Tensorflow version (2.12.0).

## New in Release 2.5.0

Post-training Quantization:

- Features:
  - Official release of OpenVINO framework support.
    - Ported NNCF OpenVINO backend to use the [nGraph](https://docs.openvino.ai/2021.3/openvino_docs_nGraph_DG_Introduction.html) representation of OpenVINO models.
    - Changed dependencies of NNCF OpenVINO backend. It now depends on `openvino` package and not on the `openvino-dev` package.
    - Added GRU/LSTM quantization support.
    - Added quantizer scales unification.
    - Added support for models with 3D and 5D Depthwise convolution.
    - Added FP16 OpenVINO models support.
  - Added `"overflow_fix"` parameter (for `quantize(...)` & `quantize_with_accuracy_control(...)` methods) support & functionality. It improves accuracy for optimized model for affected devices. More details in [Quantization section](docs/usage/post_training_compression/post_training_quantization/Usage.md).
  - (OpenVINO) Added support for in-place statistics collection (reduce memory footprint during optimization).
  - (OpenVINO) Added Quantization with accuracy control algorithm.
  - (OpenVINO) Added YOLOv8 examples for [`quantize(...)`](examples/post_training_quantization/openvino/yolov8) & [`quantize_with_accuracy_control(...)`](examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control) methods.
  - (PyTorch) Added min-max quantization algorithm as experimental.

- Fixes:
  - Fixed `ignored_scope` attribute behaviour for weights. Now, the weighted layers excludes from optimization scope correctly.
  - (ONNX) Checking correct ONNX opset version via the `nncf.quantize(...)`. Now, models with opset < 13 are optimized correctly in per-tensor quantization.

- Improvements:
  - Added improvements for statistic collection process (collect weights statistics only once).
  - (PyTorch, OpenVINO, ONNX) Introduced unified quantizer parameters calculation.

- Known issues:
  - `quantize(...)` method can generate inaccurate int8 results for models with the *DenseNet-like* architecture. Use `quantize_with_accuracy_control(...)` in such case.
  - `quantize(...)` method can hang on models with *transformer* architecture when `fast_bias_correction` optional parameter is set to *False*. Don't set it to *False* or use `quantize_with_accuracy_control(...)` in such case.
  - `quantize(...)` method can generate inaccurate int8 results for models with the *MobileNet-like* architecture on non-VNNI machines.

Compression-aware training:

- New Features:
  - Introduced automated structured pruning algorithm for JPQD with support for BERT, Wave2VecV2, Swin, ViT, DistilBERT, CLIP, and MobileBERT models.
  - Added `nncf.common.utils.patcher.Patcher` - this class can be used to patch methods on live PyTorch model objects with wrappers such as `nncf.torch.dynamic_graph.context.no_nncf_trace` when doing so in the model code is not possible (e.g. if the model comes from an external library package).
  - Compression controllers of the `nncf.api.compression.CompressionAlgorithmController` class now have a `.strip()` method that will return the compressed model object with as many custom NNCF additions removed as possible while preserving the functioning of the model object as a compressed model.

- Fixes:
  - Fixed statistics computation for pruned layers.
  - (PyTorch) Fixed traced tensors to implement the YOLOv8 from Ultralytics.

- Improvements:
  - Extension of attributes (`transpose/permute/getitem`) for pruning node selector.
  - NNCFNetwork was refactored from a wrapper-approach to a mixin-like approach.
  - Added average pool 3d-like ops to pruning mask.
  - Added Conv3d for overflow fix.
  - `nncf.set_log_file(...)` can now be used to set location of the NNCF log file.
  - (PyTorch) Added support for pruning of `torch.nn.functional.pad` operation.
  - (PyTorch) Added `torch.baddbmm` as an alias for the matmul metatype for quantization purposes.
  - (PyTorch) Added config file for ResNet18 accuracy-aware pruning + quantization on CIFAR10.
  - (PyTorch) Fixed JIT-traceable PyTorch models with internal patching.
  - (PyTorch) Added `__matmul__` magic functions to the list of patched ops (for SwinTransformer by Microsoft).

- Requirements:
  - Updated ONNX version (1.13)
  - Updated Tensorflow version (2.11)

- General changes:
  - Added Windows support for NNCF.

## New in Release 2.4.0

Target version updates:

- Bump target framework versions to PyTorch 1.13.1, TensorFlow 2.8.x, ONNX 1.12, ONNXRuntime 1.13.1
- Increased target HuggingFace transformers version for the integration patch to 4.23.1

Features:

- Official release of the ONNX framework support.
NNCF may now be used for post-training quantization (PTQ) on ONNX models.
Added an [example script](examples/post_training_quantization/onnx/mobilenet_v2) demonstrating the ONNX post-training quantization on MobileNetV2.
- Preview release of OpenVINO framework support.
NNCF may now be used for post-training quantization on OpenVINO models. Added an example script demonstrating the OpenVINO post-training quantization on MobileNetV2.
`pip install nncf[openvino]` will install NNCF with the required OV framework dependencies.
- Common post-training quantization API across the supported framework model formats (PyTorch, TensorFlow, ONNX, OpenVINO IR) via the `nncf.quantize(...)` function.
The parameter set of the function is the same for all frameworks - actual framework-specific implementations are being dispatched based on the type of the model object argument.
- (PyTorch, TensorFlow) Improved the adaptive compression training functionality to reduce effective training time.
- (ONNX) Post-processing nodes are now automatically excluded from quantization.
- (PyTorch - Experimental) Joint Pruning, Quantization and Distillation for Transformers enabled for certain models from HuggingFace `transformers` repo.
See [description](nncf/experimental/torch/sparsity/movement/MovementSparsity.md) of the movement pruning involved in the JPQD for details.

Bugfixes:

- Fixed a division by zero if every operation is added to ignored scope
- Improved logging output, cutting down on the number of messages being output to the standard `logging.INFO` log level.
- Fixed FLOPS calculation for linear filters - this impacts existing models that were pruned with a FLOPS target.
- "chunk" and "split" ops are correctly handled during pruning.
- Linear layers may now be pruned by input and output independently.
- Matmul-like operations and subsequent arithmetic operations are now treated as a fused pattern.
- (PyTorch) Fixed a rare condition with accumulator overflow in CUDA quantization kernels, which led to CUDA runtime errors and NaN values appearing in quantized tensors and
- (PyTorch) `transformers` integration patch now allows to export to ONNX during training, and not only at the end of it.
- (PyTorch) `torch.nn.utils.weight_norm` weights are now detected correctly.
- (PyTorch) Exporting a model with sparsity or pruning no longer leads to weights in the original model object in-memory to be hard-set to 0.
- (PyTorch - Experimental) improved automatic search of blocks to skip within the NAS algorithm – overlapping blocks are correctly filtered.
- (PyTorch, TensorFlow) Various bugs and issues with compression training were fixed.
- (TensorFlow) Fixed an error with `"num_bn_adaptation_samples": 0` in config leading to a `TypeError` during quantization algo initialization.
- (ONNX) Temporary model file is no longer saved on disk.
- (ONNX) Depthwise convolutions are now quantizable in per-channel mode.
- (ONNX) Improved the working time of PTQ by optimizing the calls to ONNX shape inferencing.

Breaking changes:

- Fused patterns will be excluded from quantization via `ignored_scopes` only if the top-most node in data flow order matches against `ignored_scopes`
- NNCF config's `"ignored_scopes"` and `"target_scopes"` are now strictly checked to be matching against at least one node in the model graph instead of silently ignoring the unmatched entries.
- Calling `setup.py` directly to install NNCF is deprecated and no longer guaranteed to work.
- Importing NNCF logger as `from nncf.common.utils.logger import logger as nncf_logger` is deprecated - use `from nncf import nncf_logger` instead.
- `pruning_rate` is renamed to `pruning_level` in pruning compression controllers.
- (ONNX) Removed CompressionBuilder. Excluded examples of NNCF for ONNX with CompressionBuilder API

## New in Release 2.3.0

- (ONNX) PTQ API support for ONNX.
- (ONNX) Added PTQ examples for ONNX in image classification, object detection, and semantic segmentation.
- (PyTorch) Added `BootstrapNAS` to find high-performing sub-networks from the super-network optimization.

Bugfixes:

- (PyTorch) Returned the initial quantized model when the retraining failed to find out the best checkpoint.
- (Experimental) Fixed weight initialization for `ONNXGraph` and `MinMaxQuantization`

## New in Release 2.2.0

- (TensorFlow) Added TensorFlow 2.5.x support.
- (TensorFlow) The `SubclassedConverter` class was added to create `NNCFGraph` for the `tf.Graph` Keras model.
- (TensorFlow) Added `TFOpLambda` layer support with `TFModelConverter`, `TFModelTransformer`, and `TFOpLambdaMetatype`.
- (TensorFlow) Patterns from `MatMul` and `Conv2D` to `BiasAdd` and `Metatypes` of TensorFlow operations with weights `TFOpWithWeightsMetatype` are added.
- (PyTorch, TensorFlow) Added prunings for `Reshape` and `Linear` as `ReshapePruningOp` and `LinearPruningOp`.
- (PyTorch) Added mixed precision quantization config with HAWQ for `Resnet50` and `Mobilenet_v2` for the latest VPU.
- (PyTorch) Splitted `NNCFBatchNorm` into `NNCFBatchNorm1d`, `NNCFBatchNorm2d`, `NNCFBatchNorm3d`.
- (PyTorch - Experimental) Added the `BNASTrainingController` and `BNASTrainingAlgorithm` for BootstrapNAS to search the model's architecture.
- (Experimental) ONNX `ModelProto` is now converted to `NNCFGraph` through `GraphConverter`.
- (Experimental) `ONNXOpMetatype` and extended patterns for fusing HW config is now available.
- (Experimental) Added `ONNXPostTrainingQuantization` and `MinMaxQuantization` supports for ONNX.

Bugfixes:

- (PyTorch, TensorFlow) Added exception handling of BN adaptation for zero sample values.
- (PyTorch, TensorFlow) Fixed learning rate after validation step for `EarlyExitCompressionTrainingLoop`.
- (PyTorch) Fixed `FakeQuantizer` to make exact zeros.
- (PyTorch) Fixed `Quantizer` misplacements during ONNX export.
- (PyTorch) Restored device information during ONNX export.
- (PyTorch) Fixed the statistics collection from the pruned model.

## New in Release 2.1.0

- (PyTorch) All PyTorch operations are now NNCF-wrapped automatically.
- (TensorFlow) Scales for concat-affecting quantizers are now unified
- (PyTorch) The pruned filters are now set to 0 in the exported ONNX file instead of removing them from the ONNX definition.
- (PyTorch, TensorFlow) Extended accuracy-aware training pipeline with the `early_exit` mode.
- (PyTorch, TensorFlow) Added support for quantization presets to be specified in NNCF config.
- (PyTorch, TensorFlow) Extended pruning statistics displayed to the user.
- (PyTorch, TensorFlow) Users may now register a `dump_checkpoints_fn` callback to control the location of checkpoint saving during accuracy-aware training.
- (PyTorch, TensorFlow) Default pruning schedule is now exponential.
- (PyTorch) SILU activation now supported.
- (PyTorch) Dynamic graph no longer traced during compressed model execution, which improves training performance of models compressed with NNCF.
- (PyTorch) Added BERT-MRPC quantization results and integration instructions to the HuggingFace Transformers integration patch.
- (PyTorch) Knowledge distillation extended with the option to specify temperature for the `softmax` mode.
- (TensorFlow) Added `mixed_min_max` option for quantizer range initialization.
- (PyTorch, TensorFlow) ReLU6-based HSwish and HSigmoid activations are now properly fused.
- (PyTorch - Experimental) Added an algorithm to search the model's architecture for basic building blocks.

Bugfixes:

- (TensorFlow) Fixed a bug where an operation with int32 inputs (following a Cast op) was attempted to be quantized.
- (PyTorch, TensorFlow) LeakyReLU now properly handled during pruning
- (PyTorch) Fixed errors with custom modules failing at the `determine_subtype` stage of metatype assignment.
- (PyTorch) Fix handling modules with `torch.nn.utils.weight_norm.WeightNorm` applied

## New in Release 2.0.2

Target version updates:

- Relax TensorFlow version requirements to 2.4.x

## New in Release 2.0.1

Target version updates:

- Bump target framework versions to PyTorch 1.9.1 and TensorFlow 2.4.3
- Increased target HuggingFace transformers version for the integration patch to 4.9.1

Bugfixes:

- (PyTorch, TensorFlow) Fixed statistic collection for the algo mixing scenario
- (PyTorch, TensorFlow) Increased pruning algorithm robustness in cases of a disconnected NNCF graph
- (PyTorch, TensorFlow) Fixed the fatality of NNCF graph PNG rendering failures
- (PyTorch, TensorFlow) Fixed README command lines
- (PyTorch) Fixed a bug with quantizing shared weights multiple times
- (PyTorch) Fixed knowledge distillation failures in CPU-only and DataParallel scenarios
- (PyTorch) Fixed sparsity application for torch.nn.Embedding and EmbeddingBag modules
- (PyTorch) Added GroupNorm + ReLU as a fusible pattern
- (TensorFlow) Fixed gamma fusion handling for pruning TF BatchNorm
- (PyTorch) Fixed pruning for models where operations have multiple convolution predecessors
- (PyTorch) Fixed NNCFNetwork wrapper so that `self` in the calls to the wrapped model refers to the wrapper NNCFNetwork object and not to the wrapped model
- (PyTorch) Fixed tracing of `view` operations to handle shape arguments with the `torch.Tensor` type
- (PyTorch) Added matmul ops to be considered for fusing
- (PyTorch, TensorFlow) Fixed tensorboard logging for accuracy-aware scenarios
- (PyTorch, TensorFlow) Fixed FLOPS calculation for grouped convolutions
- (PyTorch) Fixed knowledge distillation failures for tensors of unsupported shapes - will now ignore output tensors with unsupported shapes instead of crashing.

## New in Release 2.0

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
- (TensorFlow) Algo-mixing: Added configuration files and reference checkpoints for filter-pruned + quantized models: ResNet50@ImageNet2012(40% of filters pruned + INT8), RetinaNet@COCO2017(40% of filters pruned + INT8).
- (Experimental, PyTorch) [Learned Global Ranking](https://arxiv.org/abs/1904.12368) filter pruning mechanism for better pruning ratios with less accuracy drop for a broad range of models has been implemented.
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

## New in Release 1.7.1

Bugfixes:

- Fixed a bug with where compressed models that were supposed to return named tuples actually returned regular tuples
- Fixed an issue with batch norm adaptation-enabled compression runs hanging in the DDP scenario

## New in Release 1.7

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

## New in Release 1.6

- Added AutoQ - an AutoML-based mixed-precision initialization mode for quantization, which utilizes the power of reinforcement learning to select the best quantizer configuration for any model in terms of quality metric for a given HW architecture type.
- NNCF now supports inserting compression operations as pre-hooks to PyTorch operations, instead of abusing the post-hooking; the flexibility of quantization setups has been improved as a result of this change.
- Improved the pruning algorithm to group together dependent filters from different layers in the network and prune these together
- Extended the ONNX compressed model exporting interface with an option to explicitly name input and output tensors
- Changed the compression scheduler so that the corresponding epoch_step and step methods should now be called in the beginning of the epoch and before the optimizer step (previously these were called in the end of the epoch and after the optimizer step respectively)
- Data-dependent compression algorithm initialization is now specified in terms of dataset samples instead of training batches, e.g. `"num_init_samples"` should be used in place of "num_init_steps" in NNCF config files.
- Custom user modules to be registered for compression can now be specified to be ignored for certain compression algorithms
- Batch norm adaptation now being applied by default for all compression algorithms
- Bumped target PyTorch version to 1.7.0
- Custom OpenVINO operations such as "FakeQuantize" that appear in NNCF-exported ONNX models now have their ONNX `domain` set to org.openvinotoolkit
- The quantization algorithm will now quantize nn.Embedding and nn.EmbeddingBag weights when targeting CPU
- Added an option to optimize logarithms of quantizer scales instead of scales themselves directly, a technique which improves convergence in certain cases
- Added reference checkpoints for filter-pruned models: UNet@Mapillary (25% of filters pruned), SSD300@VOC (40% of filters pruned)

## New in Release 1.5

- Switched to using the propagation-based mode for quantizer setup by default. Compared to the previous default, pattern-based mode, the propagation-based mode better ensures that all the inputs to operations that can be quantized on a given type of hardware are quantized in accordance with what this hardware allows. Default target hardware is CPU - adjustable via `"target_device"` option in the NNCF config. More details can be found in [Quantization.md](./docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#quantizer-setup-and-hardware-config-files).
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

## New in Release 1.4

- Models with filter pruning applied are now exportable to ONNX
- BatchNorm adaptation now available as a common compression algorithm initialization step - currently disabled by default, see `"batchnorm_adaptation"` config parameters in compression algorithm documentation (e.g. [Quantizer.md](docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md)) for instructions on how to enable it in NNCF config
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

## New in Release 1.3

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

## New in Release 1.2

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
