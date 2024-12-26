[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/nncf?color=green)](https://github.com/openvinotoolkit/nncf/releases)
[![Website](https://img.shields.io/website?up_color=blue&up_message=docs&url=https%3A%2F%2Fdocs.openvino.ai%2Fnncf)](https://docs.openvino.ai/nncf)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/nncf?tab=Apache-2.0-1-ov-file#readme)
[![PyPI Downloads](https://static.pepy.tech/badge/nncf)](https://pypi.org/project/nncf/)

# Neural Network Compression Framework (NNCF)

Neural Network Compression Framework (NNCF) provides a suite of post-training
and training-time algorithms for optimizing inference of neural networks in
[OpenVINO&trade;](https://docs.openvino.ai) with a minimal accuracy drop.

NNCF is designed to work with models from [PyTorch](https://pytorch.org/),
[TensorFlow](https://www.tensorflow.org/), [ONNX](https://onnx.ai/) and
[OpenVINO&trade;](https://docs.openvino.ai).

The framework is organized as a Python package that can be built and used
as a standalone tool. Its architecture is unified to make adding different
compression algorithms easy for both PyTorch and TensorFlow.

NNCF provides samples that demonstrate the usage of compression algorithms
for different use cases and models. See compression results achievable
with the NNCF-powered samples on the
[NNCF Model Zoo page](https://github.com/openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md).

For more information about NNCF, see:

- [NNCF repository](https://github.com/openvinotoolkit/nncf)
- [User documentation](https://docs.openvino.ai/nncf)
- [NNCF API documentation](https://openvinotoolkit.github.io/nncf/autoapi/nncf/)
- [Usage examples](https://github.com/openvinotoolkit/nncf/tree/develop/docs/usage)
- [Notebook tutorials](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/README.md#model-training)

## Table of contents

- [Key Features](#key-features)
- [Installation](#installation-guide)
- [Third-party integration](#third-party-repository-integration)
- [NNCF Compressed Model Zoo](#nncf-compressed-model-zoo)

## Key Features<a id="key-features"></a>

### Post-Training Compression Algorithms

| Compression algorithm                                                                                                                                       | OpenVINO  | PyTorch   | TensorFlow    | ONNX          |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------: | :-------: | :-----------: | :-----------: |
| [Post-Training Quantization](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/post_training_quantization/Usage.md) | Supported | Supported | Supported     | Supported     |
| [Weight Compression](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)                | Supported | Supported | Not supported | Not supported |
| [Activation Sparsity](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsify_activations/ActivationSparsity.md)              | Not supported | Experimental |Not supported| Not supported |

### Training-Time Compression Algorithms

| Compression algorithm                                                                                                                                                                         | PyTorch      | TensorFlow    |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------: | :-----------: |
| [Quantization Aware Training](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/quantization_aware_training/Usage.md)                                 | Supported    | Supported     |
| [Mixed-Precision Quantization](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#mixed-precision-quantization) | Supported    | Not supported |
| [Sparsity](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Sparsity.md)                                                            | Supported    | Supported     |
| [Filter pruning](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Pruning.md)                                                       | Supported    | Supported     |
| [Movement pruning](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md)                                                        | Experimental | Not supported |

- Automatic, configurable model graph transformation to obtain the compressed
  model.
  > **NOTE**: Limited support for TensorFlow models. Only models created using
    Sequential or Keras Functional API are supported.
- Common interface for compression methods.
- GPU-accelerated layers for faster compressed model fine-tuning.
- Distributed training support.
- Git patch for prominent third-party repository
  ([huggingface-transformers](https://github.com/huggingface/transformers))
  demonstrating the process of integrating NNCF into custom training pipelines.
- Seamless combination of pruning, sparsity, and quantization algorithms. Refer
  to [optimum-intel](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino)
  for examples of joint (movement) pruning, quantization, and distillation
  (JPQD), end-to-end from NNCF optimization to compressed OpenVINO IR.
- Exporting PyTorch compressed models to ONNX\* checkpoints and TensorFlow
  compressed models to SavedModel or Frozen Graph format, ready to use with
  [OpenVINO&trade; toolkit](https://docs.openvino.ai).
- Support for [Accuracy-Aware model training](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/training_time_compression/other_algorithms/Usage.md#accuracy-aware-model-training)
  pipelines via the [Adaptive Compression Level Training](https://github.com/openvinotoolkit/nncf/blob/develop/docs/accuracy_aware_model_training/AdaptiveCompressionLevelTraining.md)
  and [Early Exit Training](https://github.com/openvinotoolkit/nncf/blob/develop/docs/accuracy_aware_model_training/EarlyExitTraining.md).

## Installation Guide<a id="installation-guide"></a>

For detailed installation instructions, refer to the [Installation](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md) guide.

NNCF can be installed as a regular PyPI package via pip:

```bash
pip install nncf
```

NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):

```bash
conda install -c conda-forge nncf
```

System requirements of NNCF correspond to the used backend. System requirements for each backend and
the matrix of corresponding versions can be found in [installation.md](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md).

## Third-party Repository Integration<a id="third-party-repository-integration"></a>

NNCF may be easily integrated into training/evaluation pipelines of third-party
repositories.

- [OpenVINO Training Extensions](https://github.com/openvinotoolkit/training_extensions)

  NNCF is integrated into OpenVINO Training Extensions as a model optimization
  backend. You can train, optimize, and export new models based on available
  model templates as well as run the exported models with OpenVINO.

- [HuggingFace Optimum Intel](https://huggingface.co/docs/optimum/intel/optimization_ov)

  NNCF is used as a compression backend within the renowned `transformers`
  repository in HuggingFace Optimum Intel.

## NNCF Compressed Model Zoo<a id="nncf-compressed-model-zoo"></a>

A list of models and compression results for them can be found at our
[NNCF Model Zoo page](https://github.com/openvinotoolkit/nncf/blob/develop/docs/ModelZoo.md).

## Citing

```bi
@article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov, Alexander and Lazarevich, Ivan and Shamporov, Vasily and Lyalyushkin, Nikolay and Gorbachev, Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
}
```

## Telemetry

NNCF as part of the OpenVINO™ toolkit collects anonymous usage data for the
purpose of improving OpenVINO™ tools. You can opt-out at any time by running
the following command in the Python environment where you have NNCF installed:

`opt_in_out --opt_out`

More information available on [OpenVINO telemetry](https://docs.openvino.ai/nightly/about-openvino/additional-resources/telemetry.html).
