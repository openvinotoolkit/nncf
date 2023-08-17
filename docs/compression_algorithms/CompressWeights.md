### Weights Compression

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run Weights Compression with, and PyTorch is also supported.

#### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights only for Linear and Embedding layers.

#### User guide

- Compress weights of linear layers and embeddings to int8

```python
from nncf import compress_weights
compressed_model = compress_weights(model)
```
##### Limitations

- The algorithm is supported for OpenVINO and PyTorch models.
- The compression applies in-place.
- The compressed model is not trainable.
