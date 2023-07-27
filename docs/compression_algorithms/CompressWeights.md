### Compress Weights

#### The algorithm description

The Weights Compression algorithm compresses weights in LLM to int8 and adds extra decompression layers to the input network or keeps original weights precision and inserts FakeQuantize operations. The goal is to compress weights of big LLMs before conversion to IR without usage of dataset. The algorithm compresses weights only for Linear and Embedding layers.

#### User guide

```python
from nncf import compress_weights
compressed_model = compress_weights(model)
```

```python
from nncf import compress_weights
model_with_fake_quantize = compress_weights(model, True)
```

##### Limitations

- The algorithm is supported for PyTorch only.
- The compression applies in-place.
- The compressed model is not trainable.
