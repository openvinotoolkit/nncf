### Compress Weights

#### The algorithm description

The Weights Compression algorithm compresses weights in LLM to int8 and adds extra decompression layers to the input network. The goal is to compress weights of big LLMs before conversion to IR without usage of dataset. The algorithm compresses weights only for Linear and Embedding layers. It is also possible to keeps original weights precision and inserts FakeQuantize operations by setting use_fake_quantize parameter to True.

#### User guide

- Compress weights of linear layers and embeddings to int8

```python
from nncf import compress_weights
compressed_model = compress_weights(model)
```

- Insert FakeQuantize layers for weights of linear layers and embeddings

```python
from nncf import compress_weights
model_with_fake_quantize = compress_weights(model, use_fake_quantize=True)
```

##### Limitations

- The algorithm is supported for PyTorch only.
- The compression applies in-place.
- The compressed model is not trainable.
