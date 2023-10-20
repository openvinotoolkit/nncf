### Weights Compression

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run Weights Compression with, and PyTorch is also supported.

#### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights only for Linear and Embedding layers.

##### INT8 and NF4 modes

By default, weights are compressed to 8-bit integer data type - "INT8" mode.
OpenVINO backend has also an experimental support for "NF4" mode - compression to [nf4](https://arxiv.org/pdf/2305.14314v1.pdf) data type.
It goes with a grouped quantization, when small group of weights (e.g. 128) in the channel dimension share quantization parameters (scale).
First embedding and last linear layers are always compressed to 8-bit integer data type in the "NF4" mode.
Percent of the rest layers compressed to NF4 can be configured by "ratio" parameter.
E.g. ratio=0.9 means 90% of layers compressed to nf4 and the rest to 8-bit integer data type.

#### User guide

- Compress weights to 8-bit integer data type.

```python
from nncf import compress_weights
compressed_model = compress_weights(model)
```

- Compress weights to nf4 data type with group size = 128, except first embedding and last linear layers - they are compressed to 8-bit integer data type.

```python
from nncf import compress_weights
from nncf import CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4)
```

- Compress weights of 90% of layers to nf4 with the group size 64, and the rest of layers to 8-bit integer data type.

```python
from nncf import compress_weights
from nncf import CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4, group_size=64, ratio=0.9)
```

##### Limitations

- The algorithm is supported for OpenVINO and PyTorch models.
- The compression applies in-place.
- The compressed model is not trainable.
- NF4 mode, grouped quantization and mixed nf4-int8 precision selection is available for OpenVINO backend only.
- NF4 support is experimental - models quantized to nf4 should not be faster models quantized to 8-bit integer.
