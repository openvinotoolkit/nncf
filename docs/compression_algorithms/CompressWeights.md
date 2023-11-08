### Weights Compression

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run Weights Compression with, and PyTorch is also supported.

#### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights only for Linear and Embedding layers.

#### Supported modes

By default, weights are compressed to 8-bit integer data type - "INT8" mode.
OpenVINO backend also supports 3 modes of mixed precision weight quantization with a 4-bit data type as a primary precision - INT4_SYM, INT4_ASYM and NF4. The primary precision in case of INT4_SYM mode is unsigned 4-bit integer and weights are quantized to it [symmetrically](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization) with a fixed zero point equals to 8. In case of INT4_ASYM mode - also unsigned 4-bit integer, but weight are quantized to it [asymmetrically](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization) with a typical non-fixed zero point. In case of NF4 mode - [nf4](https://arxiv.org/pdf/2305.14314v1.pdf) data type without zero point.
All 4-bit modes have a grouped quantization support, when small group of weights (e.g. 128) in the channel dimension share quantization parameters (scale).
First embedding and last linear layers are always compressed to 8-bit integer data type.
Percent of the rest layers compressed to 4-bit can be configured by "ratio" parameter. E.g. ratio=0.9 means 90% of layers compressed to the corresponding 4-bit data type and the rest to 8-bit integer data type.

#### User guide

- Compress weights to 8-bit integer data type.

```python
from nncf import compress_weights
compressed_model = compress_weights(model)
```

- Compress weights symmetrically to 4-bit integer data type with group size = 128, except first embedding and last linear layers - they are compressed to 8-bit integer data type.

```python
from nncf import compress_weights
from nncf import CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM)
```

- Generally, `INT4_SYM` mode is the fastest mixed-precision mode, but it may lead to a significant accuracy degradation or perplexity increase.
  Compressing weights asymmetrically (`INT4_ASYM` mode) is the way to increase accuracy, however in turns it slows down inference a bit.
  If the accuracy or perplexity is still not satisfying, there are 2 more hyper-parameters to tune: `group_size` and `ratio`.
  Lower group size and less ratio of 4-bit layers usually improve accuracy at the sacrifice of inference speed.
  Below is the example how to compress weights of 90% of layers to 4-bit integer asymmetrically with the group size 64, and
  the rest of layers to 8-bit integer data type. The same parametrization is applicable for `INT4_SYM` mode.

```python
from nncf import compress_weights
from nncf import CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, group_size=64, ratio=0.9)
```

- `NF4` mode can be considered for improving accuracy, but currently models quantized to nf4 should not be faster models
  quantized to 8-bit integer. Here's the example how to compress weights to nf4 data type with group size = 128.
  Different `group_size` and `ratio` are also supported.

```python
from nncf import compress_weights
from nncf import CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4)
```

#### Evaluation results

Here is the perplexity and model size before and after weight compression for different language models on the [Lambada OpenAI dataset](https://github.com/openai/gpt-2/issues/131#issuecomment-497136199).
`g32` refers to the group size equals to 32, `r60` - to the ratio equals to 0.6.

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
    <td class="tg-0pky">int4_sym_g64_r80</td>
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

#### Limitations

- The algorithm is supported for OpenVINO and PyTorch models.
- The compression applies in-place.
- The compressed model is not trainable.
- INT4_SYM, INT4_ASYM and NF4 modes, grouped quantization and mixed precision selection is available for OpenVINO backend only.
- NF4 support is experimental - models quantized to nf4 should not be faster models quantized to 8-bit integer.

#### Additional resources

- [LLM Weight Compression](https://docs.openvino.ai/nightly/weight_compression.html)
- [Optimize and Deploy Generative AI Models using Hugging Face Optimum Intel](https://docs.openvino.ai/nightly/gen_ai_guide.html)
- [Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference)
