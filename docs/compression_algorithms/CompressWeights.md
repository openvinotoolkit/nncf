### Weights Compression

#### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights only for Linear and Embedding layers. It is also possible to keep the precision of the original weights and insert FakeQuantize operations by setting `use_fake_quantize` parameter to `True`.

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
