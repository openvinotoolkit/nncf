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

<table>
<thead>
  <tr>
    <th class="tg-0pky">Model</th>
    <th class="tg-0pky">Mode</th>
    <th class="tg-0pky">Perplexity</th>
    <th class="tg-0pky">Perplexity <br>Increase</th>
    <th class="tg-0pky">Model Size <br>(Gb)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">databricks/dolly-v2-3b</td>
    <td class="tg-0pky">fp32</td>
    <td class="tg-0pky">5.01</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">10.3</td>
  </tr>
  <tr>
    <td class="tg-0pky">databricks/dolly-v2-3b</td>
    <td class="tg-0pky">int8</td>
    <td class="tg-0pky">5.07</td>
    <td class="tg-0pky">0.05</td>
    <td class="tg-0pky">2.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">databricks/dolly-v2-3b</td>
    <td class="tg-0pky">int4_asym_g32_r50</td>
    <td class="tg-0pky">5.28</td>
    <td class="tg-0pky">0.26</td>
    <td class="tg-0pky">2.2</td>
  </tr>
  <tr>
    <td class="tg-0pky">databricks/dolly-v2-3b</td>
    <td class="tg-0pky">nf4_g128_r60</td>
    <td class="tg-0pky">5.19</td>
    <td class="tg-0pky">0.18</td>
    <td class="tg-0pky">1.9</td>
  </tr>
  <tr>
    <td class="tg-0pky">facebook/opt-6.7b</td>
    <td class="tg-0pky">fp32</td>
    <td class="tg-0pky">4.25</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">24.8</td>
  </tr>
  <tr>
    <td class="tg-0pky">facebook/opt-6.7b</td>
    <td class="tg-0pky">int8</td>
    <td class="tg-0pky">4.27</td>
    <td class="tg-0pky">0.01</td>
    <td class="tg-0pky">6.2</td>
  </tr>
  <tr>
    <td class="tg-0pky">facebook/opt-6.7b</td>
    <td class="tg-0pky">int4_asym_g64_r80</td>
    <td class="tg-0pky">4.32</td>
    <td class="tg-0pky">0.07</td>
    <td class="tg-0pky">4.1</td>
  </tr>
  <tr>
    <td class="tg-0pky">facebook/opt-6.7b</td>
    <td class="tg-0pky">nf4_g64</td>
    <td class="tg-0pky">4.35</td>
    <td class="tg-0pky">0.1</td>
    <td class="tg-0pky">3.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-7b-chat-hf</td>
    <td class="tg-0pky">fp32</td>
    <td class="tg-0pky">3.28</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">25.1</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-7b-chat-hf</td>
    <td class="tg-0pky">int8</td>
    <td class="tg-0pky">3.29</td>
    <td class="tg-0pky">0.01</td>
    <td class="tg-0pky">6.3</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-7b-chat-hf</td>
    <td class="tg-0pky">int4_asym_g128_r80</td>
    <td class="tg-0pky">3.41</td>
    <td class="tg-0pky">0.14</td>
    <td class="tg-0pky">4.0</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-7b-chat-hf</td>
    <td class="tg-0pky">nf4_g128</td>
    <td class="tg-0pky">3.41</td>
    <td class="tg-0pky">0.13</td>
    <td class="tg-0pky">3.5</td>
  </tr>
  <tr>
    <td class="tg-0pky">togethercomputer/RedPajama-INCITE-7B-Instruct</td>
    <td class="tg-0pky">fp32</td>
    <td class="tg-0pky">4.15</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">25.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">togethercomputer/RedPajama-INCITE-7B-Instruct</td>
    <td class="tg-0pky">int8</td>
    <td class="tg-0pky">4.17</td>
    <td class="tg-0pky">0.02</td>
    <td class="tg-0pky">6.4</td>
  </tr>
  <tr>
    <td class="tg-0pky">togethercomputer/RedPajama-INCITE-7B-Instruct</td>
    <td class="tg-0pky">nf4_ov_g32_r60</td>
    <td class="tg-0pky">4.28</td>
    <td class="tg-0pky">0.13</td>
    <td class="tg-0pky">5.1</td>
  </tr>
  <tr>
    <td class="tg-0pky">togethercomputer/RedPajama-INCITE-7B-Instruct</td>
    <td class="tg-0pky">int4_asym_g128</td>
    <td class="tg-0pky">4.17</td>
    <td class="tg-0pky">0.02</td>
    <td class="tg-0pky">3.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-13b-chat-hf</td>
    <td class="tg-0pky">fp32</td>
    <td class="tg-0pky">2.92</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">48.5</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-13b-chat-hf</td>
    <td class="tg-0pky">int8</td>
    <td class="tg-0pky">2.91</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">12.1</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-13b-chat-hf</td>
    <td class="tg-0pky">int4_asym_sym_g64_r80</td>
    <td class="tg-0pky">2.98</td>
    <td class="tg-0pky">0.06</td>
    <td class="tg-0pky">8.0</td>
  </tr>
  <tr>
    <td class="tg-0pky">meta-llama/Llama-2-13b-chat-hf</td>
    <td class="tg-0pky">nf4_g128</td>
    <td class="tg-0pky">2.95</td>
    <td class="tg-0pky">0.04</td>
    <td class="tg-0pky">6.6</td>
  </tr>
</tbody>
</table>

##### Limitations

- The algorithm is supported for OpenVINO and PyTorch models.
- The compression applies in-place.
- The compressed model is not trainable.
- NF4 mode, grouped quantization and mixed nf4-int8 precision selection is available for OpenVINO backend only.
- NF4 support is experimental - models quantized to nf4 should not be faster models quantized to 8-bit integer.
