# Implemented Compression Methods

## Post-training Compression

- [Post Training Quantization (PTQ)](./usage/post_training_compression/post_training_quantization/Usage.md) (OpenVINO, PyTorch, TorchFX, ONNX, TensorFlow)
  - Symmetric and asymmetric quantization modes
  - Signed and unsigned
  - Per tensor/per channel
  - Each backend support export to the OpenVINO format
- [Weights compression](./usage/post_training_compression/weights_compression/Usage.md) (OpenVINO, PyTorch, TorchFX)
  - Symmetric 8 bit compression mode
  - Symmetric and asymmetric 4 bit compression mode
  - NF4 compression mode
  - E2M1 weights with E8M0 scales compression mode
  - Mixed precision weights compression
  - Grouped weights compression

## Training Time Compression

- [Quantization Aware Training (QAT)](./usage/training_time_compression/quantization_aware_training/Usage.md) (PyTorch)
  - Training of a quantized model after the Post Training Quantization
  - Symmetric and asymmetric quantization modes
  - Signed and unsigned
  - Per tensor/per channel
  - Exports to OpenVINO format

### Other algorithms

Each compression method in this section receives its own hyperparameters that are organized as a dictionary and basically stored in a JSON file that is deserialized when the training starts. Compression methods can be applied separately or together producing sparse, quantized, or both sparse and quantized models. For more information about the configuration, refer to the samples.

- [Legacy Quantization Aware Training (QAT)](./usage/training_time_compression/other_algorithms/LegacyQuantization.md) (PyTorch, TensorFlow)
  - Symmetric and asymmetric quantization modes
  - Signed and unsigned
  - Per tensor/per channel
  - Exports to OpenVINO-supported FakeQuantize ONNX nodes
  - Arbitrary bitwidth
  - Mixed-bitwidth quantization
  - Automatic bitwidth assignment based on HAWQ
  - Automatic quantization parameter selection and activation quantizer setup based on HW config preset
  - Automatic bitwidth assignment mode AutoQ, based on HAQ, a Deep Reinforcement Learning algorithm to select best mixed precision given quality metric and HW type.
- [Unstructured sparsity](./usage/training_time_compression/other_algorithms/Sparsity.md) (PyTorch, TensorFlow)
  - Magnitude sparsity
  - Regularization-based (RB) sparsity
- [Filter pruning (Structured sparsity)](./usage/training_time_compression/other_algorithms/Pruning.md) (PyTorch, TensorFlow)
