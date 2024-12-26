
- [The algorithm description](#the-algorithm-description)
- [Supported modes](#supported-modes)
- [User guide](#user-guide)
  - [Data-free methods](#data-free-methods)
  - [Data-aware methods](#data-aware-methods)
  - [Caching Statistics](#caching-statistics)
- [Evaluation results](#evaluation-results)
  - [Data-free Mixed-Precision on Lambada OpenAI dataset](#data-free-mixed-precision-on-lambada-openai-dataset)
  - [Data-aware Mixed-Precision and AWQ methods on Wikitext dataset](#data-aware-mixed-precision-and-awq-methods-on-wikitext-dataset)
  - [Scale Estimation and GPTQ methods on Lambada OpenAI dataset](#scale-estimation-and-gptq-methods-on-lambada-openai-dataset)
  - [Accuracy/Footprint trade-off](#accuracyfootprint-trade-off)
- [Limitations](#limitations)
- [Additional resources](#additional-resources)

### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights for Linear, Convolution and Embedding layers.

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run Weights Compression with. PyTorch and Torch FX are also supported.

### Supported modes

By default, weights are compressed asymmetrically to 8-bit integer data type - "INT8_ASYM" mode.
OpenVINO backend also supports 4 modes of mixed precision weight quantization with a 4-bit data type as a primary precision - INT4_SYM, INT4_ASYM, NF4, E2M1. The primary precision in case of INT4_SYM mode is signed 4-bit integer and weights are quantized to it [symmetrically](/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#symmetric-quantization) without zero point. In case of INT4_ASYM mode - unsigned 4-bit integer and weight are quantized to it [asymmetrically](/docs/usage/training_time_compression/other_algorithms/LegacyQuantization.md#asymmetric-quantization) with a typical non-fixed zero point. In case of NF4 mode - [nf4](https://arxiv.org/pdf/2305.14314v1.pdf) data type without zero point. In case of E2M1 mode - [e2m1](https://arxiv.org/pdf/2310.10537) data type without zero point and has 8bit [E8M0](https://arxiv.org/pdf/2310.10537) scale.
All 4-bit modes have a grouped quantization support, when small group of weights (e.g. 128) in the channel dimension share quantization parameters (scale).
All embeddings, convolutions and last linear layers are always compressed to a backup mode, which is "INT8_ASYM", by default. To quantize embeddings and last linear layers to 4-bit, use `all_layers=True`.
Percent of the rest layers compressed to 4-bit can be configured by "ratio" parameter. E.g. ratio=0.9 means 90% of layers compressed to the corresponding 4-bit data type and the rest to a backup mode. OpenVINO backend supports 3 backup modes: INT8_SYM, INT8_ASYM, and NONE, which retains the original floating-point precision of the model weights. Backup mode is supported only for mixed-precision weight quantization.

### User guide

#### Data-free methods

- Compress weights asymmetrically to 8-bit integer data type.

```python
from nncf import compress_weights
compressed_model = compress_weights(model) # model is openvino.Model object
```

- Compress weights symmetrically to 8-bit integer data type.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT8_SYM) # model is openvino.Model object
```

- Compress weights symmetrically to 4-bit integer data type with group size = 128, except embeddings, convolutions and last linear layers - they are compressed asymmetrically to 8-bit integer data type.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM) # model is openvino.Model object
```

- Compress weights to NF4 with group size = 128, except embeddings, convolutions and last linear layers - they are remain in original floating-point precision.

```python
from nncf import compress_weights, BackupMode, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4, backup_mode=BackupMode.NONE) # model is openvino.Model object
```

- Generally, `INT4_SYM` mode is the fastest mixed-precision mode, but it may lead to a significant accuracy degradation or perplexity increase.
  Compressing weights asymmetrically (`INT4_ASYM` mode) is the way to increase accuracy, however in turns it slows down inference a bit.
  If the accuracy or perplexity is still not satisfying, there are 2 more hyper-parameters to tune: `group_size` and `ratio`. Please refer to the [example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams) how to automatically tune these parameters.
  Lower group size and less ratio of 4-bit layers usually improve accuracy at the sacrifice of inference speed. To disable grouped quantization and quantize weights per-channel, set `group_size = -1`.
  Below is the example how to compress weights of 90% of layers to 4-bit integer asymmetrically with the group size 64, and
  the rest of layers to 8-bit asymmetric integer data type. The same parametrization is applicable for `INT4_SYM` mode.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, group_size=64, ratio=0.9) # model is openvino.Model object
```

#### Data-aware methods

- Accuracy of the 4-bit compressed models can be improved by using data-aware mixed-precision algorithm. It is capable to find outliers in the input activations and assign different quantization precision to minimize accuracy degradation.
Below is the example how to compress 80% of layers to 4-bit integer with a default data-aware mixed precision algorithm.
It requires just one extra parameter - a NNCF wrapper of the dataset. Refer to the [full example](https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino) of data-aware weight compression for more details. If dataset is not specified, data-free mixed precision algorithm works based on weights only.
Refer to the second table below for evaluation of data-free and data-aware method on the wikitext dataset.
On the average the data-aware mixed-precision weight compression takes more time than the data-free one (~30% slower on Intel(R) Xeon(R) Gold 6430L), since it infers model on calibration dataset to find outliers in the input activations.

```python
from nncf import compress_weights, CompressWeightsMode, Dataset
nncf_dataset = nncf.Dataset(data_source, transform_fn)
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM, ratio=0.8, dataset=nncf_dataset) # model is openvino.Model object
```

- Additionally, it is possible to generate a synthetic dataset by the `nncf.data.generate_text_data` method to use it in the data-aware weight compression. The method takes a language model (e.g. from `optimum.intel.openvino`) and a tokenizer (e.g. from `transformers`) as input and returns the list of strings generated by the model. Note that the dataset generation takes time and depends on various conditions like the model size, requested dataset length or environment setup. Also, since the dataset is generated by the model output, it does not guarantee significant accuracy improvement after the compression. This method is recommended only in cases when a better dataset is not available. Refer to the [example](https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_synthetic_data/) for details of the usage.

```python
from nncf import compress_weights, CompressWeightsMode, Dataset
from nncf.data import generate_text_data
synthetic_data = generate_text_data(model, tokenizer)
nncf_dataset = nncf.Dataset(synthetic_data, transform_fn)
```

- Accuracy of the 4-bit compressed models also can be improved by using AWQ, Scale Estimation, GPTQ or Lora Correction algorithms over data-based mixed-precision algorithm. These algorithms work by equalizing a subset of weights to minimize the difference between the original precision and the 4-bit precision.
Unlike all others, the Lora Correction algorithm inserts an additional Linear layers for reducing quantization noise and further accuracy improvement. Inevitably, this approach introduces a memory and a runtime overheads, but they are negligible, since the inserted weight much smaller and can be quantized to 8-bit. The AWQ, Scale Estimation (SE) and Lora Correction (LC) algo can be used in any combination together: AWQ + SE, AWQ + LC, SE + LC, AWQ + SE + LC. The GPTQ algorithm can be combined with AWQ and Scale Estimation in any combination: AWQ + GPTQ, GPTQ + SE, AWQ + GPTQ + SE. Below are examples demonstrating how to enable the AWQ, Scale Estimation, GPTQ or Lora Correction algorithms:

  <details>
  <summary>Prepare the calibration dataset for data-based algorithms</summary>

  ```python
  from datasets import load_dataset
  from functools import partial
  from nncf import compress_weights, CompressWeightsMode, Dataset
  from optimum.intel.openvino import OVModelForCausalLM
  from transformers import AutoTokenizer

  def transform_func(item, tokenizer, input_shapes):
      text = item['text']
      tokens = tokenizer(text)

      res = {'input_ids': np.expand_dims(np.array(tokens['input_ids']), 0),
            'attention_mask': np.expand_dims(np.array(tokens['attention_mask']), 0)}

      if 'position_ids' in input_shapes:
          position_ids = np.cumsum(res['attention_mask'], axis=1) - 1
          position_ids[res['attention_mask'] == 0] = 1
          res['position_ids'] = position_ids

      for name, shape in input_shapes.items():
          if name in res:
              continue
          res[name] = np.zeros(shape)

      return res

  def get_input_shapes(model, batch_size = 1):
      inputs = {}

      for val in model.model.inputs:
          name = val.any_name
          shape = list(val.partial_shape.get_min_shape())
          shape[0] = batch_size
          inputs[name] = shape

      return inputs

  # load your model and tokenizer
  model = OVModelForCausalLM.from_pretrained(...)
  tokenizer = AutoTokenizer.from_pretrained(...)

  # prepare dataset for compression
  dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train')
  dataset = dataset.filter(lambda example: len(example["text"]) > 80)
  input_shapes = get_input_shapes(model)
  nncf_dataset = Dataset(dataset, partial(transform_func, tokenizer=tokenizer,
                                                          input_shapes=input_shapes))
  ```

  </details>

- How to compress 80% of layers to 4-bit integer with a default data-based mixed precision algorithm and AWQ with Scale Estimation. It requires to set `awq` to `True` and `scale_estimation` to `True` additionally to data-based mixed-precision algorithm.

```python
model.model = compress_weights(model.model,
                               mode=CompressWeightsMode.INT4_SYM,
                               ratio=0.8,
                               dataset=nncf_dataset,
                               awq=True,
                               scale_estimation=True)
```

- How to compress 80% of layers to 4-bit integer with a default data-based mixed precision algorithm and GPTQ. It requires to set `gptq` to `True` additionally to data-based mixed-precision algorithm.

```python
model.model = compress_weights(model.model,
                               mode=CompressWeightsMode.INT4_SYM,
                               ratio=0.8,
                               dataset=nncf_dataset,
                               gptq=True)
```

- How to compress 80% of layers to 4-bit integer with a default data-based mixed precision algorithm and Lora Correction algorithm. It requires setting `lora_correction` to `True` additionally to data-based mixed-precision algorithm.

```python
model.model = compress_weights(model.model,
                               mode=CompressWeightsMode.INT4_SYM,
                               ratio=0.8,
                               dataset=nncf_dataset,
                               lora_correction=True)
```

- `NF4` mode can be considered for improving accuracy, but currently models quantized to nf4 should not be faster models
  quantized to 8-bit asymmetric integer. Here's the example how to compress weights to nf4 data type with group size = 128.
  Different `group_size` and `ratio` are also supported.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4)
```

- `E2M1` mode can be considered for improving accuracy, but currently models quantized to e2m1 should not be faster models
  quantized to 8-bit asymmetric integer. Here's the example how to compress weights to e2m1 data type with group size = 32 (recommended).
  Different `group_size` and `ratio` are also supported.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.E2M1, group_size=32, all_layers=True)
```

#### Caching Statistics

To optimize compression time and reuse statistics across multiple configurations, you can use the `statistics_path` option. This feature enables caching of calculated statistics, allowing them to be loaded from a specified path rather than recalculated for each configuration. This approach can significantly reduce compression time during repeated model compression iterations, making it ideal when searching for optimal compression parameters.

To enable statistics caching, set the `statistics_path` parameter to your chosen path.

```python
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf import compress_weights

compressed_model = compress_weights(
    model,
    advanced_parameters=AdvancedCompressionParameters(statistics_path="statistics")
)
```

When `statistics_path` is provided, the system first checks if the specified path exists. If it does, the statistics are loaded from this path. If the path does not exist, the statistics are computed and saved to this path for future use.

### Evaluation results

#### Data-free Mixed-Precision on Lambada OpenAI dataset

Here is the perplexity and model size before and after weight compression for different language models on the [Lambada OpenAI dataset](https://github.com/openai/gpt-2/issues/131#issuecomment-497136199).
`g32` refers to the group size equals to 32, `r60` - to the ratio equals to 0.6.

<table>
<thead>
  <tr>
    <th class="tg-0pky">Model</th>
    <th class="tg-0pky">Mode</th>
    <th class="tg-0pky">Perplexity (↓)</th>
    <th class="tg-0pky">Perplexity <br>Increase (↓)</th>
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
    <td class="tg-0pky">int8_asym</td>
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
    <td class="tg-0pky">int8_asym</td>
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
    <td class="tg-0pky">int8_asym</td>
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
    <td class="tg-0pky">int8_asym</td>
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
    <td class="tg-0pky">int8_asym</td>
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

#### Data-aware Mixed-Precision and AWQ methods on Wikitext dataset

Here is the word perplexity with data-free and data-aware mixed-precision INT4-INT8 weight compression for different language models on the [wikitext dataset](https://arxiv.org/pdf/1609.07843.pdf).
`data` suffix refers to the data-aware mixed-precision.
`data_awq` suffix refers to the data-aware mixed-precision with modified [AWQ](https://arxiv.org/abs/2306.00978) algorithm.
This modification applies only for patterns `MatMul-Multiply-MatMul` (for example MLP block in LLama).
<table>
    <tr bgcolor='#B4B5BB'>
        <td>Model</td>
        <td>Mode</td>
        <td>Word Perplexity (↓)</td>
    </tr>
        <tr>
        <td>meta-llama/llama-7b-chat-hf</td>
        <td>fp16</td>
        <td>11.57</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128_r80_data</td>
        <td>11.87</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128_r80</td>
        <td>11.92</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128_r100_data_awq</td>
        <td>12.34</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128_r100</td>
        <td>12.35</td>
    </tr>
    <tr>
        <td>stabilityai_stablelm-3b-4e1t</td>
        <td>fp16</td>
        <td>10.16</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r80_data</td>
        <td>10.67</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r80</td>
        <td>10.83</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r100_data_awq</td>
        <td>10.89</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r100</td>
        <td>11.07</td>
    </tr>
    <tr>
        <td>stable-zephyr-3b-dpo</td>
        <td>int4_sym_g64_r80_data_awq</td>
        <td>21.62</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r80_data</td>
        <td>21.74</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r80</td>
        <td>23.10</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r100_data_awq</td>
        <td>21.76</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g64_r100</td>
        <td>23.19</td>
    </tr>
    <tr>
        <td>HuggingFaceH4/zephyr-7b-beta</td>
        <td>fp16</td>
        <td>9.82</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128_r80_data</td>
        <td>10.13</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_g128</td>
        <td>10.22</td>
    </tr>
</table>

#### Scale Estimation and GPTQ methods on Lambada OpenAI dataset

Here is the perplexity and accuracy with data-free and data-aware mixed-precision INT4-INT8 weight compression for different language models on the [lambada openai dataset](https://huggingface.co/datasets/EleutherAI/lambada_openai).
`_scale` suffix refers to the data-aware mixed-precision with Scale Estimation algorithm. `_gptq` suffix refers to the data-aware mixed-precision with GPTQ algorithm. `_gptq_scale` suffix refers to the use of GPTQ algorithm with the Scale estimation algorithm to calculate the quantization parameters.
`r100` means that embeddings and lm_head have INT8 precision and all other linear layers have INT4 precision.
<table>
    <tr bgcolor='#B4B5BB'>
        <td>Model</td>
        <td>Mode</td>
        <td>Acc (↑)</td>
        <td>Ppl (↓)</td>
    </tr>
    <tr>
        <td>stabilityai_stablelm-2-zephyr-1_6b</td>
        <td>fp32</td>
        <td>0.5925</td>
        <td>6.3024</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_gptq_scale</td>
        <td>0.5795</td>
        <td>7.1507</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_gptq</td>
        <td>0.5676</td>
        <td>7.2391</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_scale</td>
        <td>0.5795</td>
        <td>7.3245</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64</td>
        <td>0.5465</td>
        <td>8.649</td>
    </tr>
    <tr>
        <td>stable-zephyr-3b-dpo</td>
        <td>fp32</td>
        <td>0.6099</td>
        <td>6.7151</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_scale</td>
        <td>0.595</td>
        <td>7.037</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_gptq_scale</td>
        <td>0.5909</td>
        <td>7.391</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64_gptq</td>
        <td>0.567</td>
        <td>8.6787</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs64</td>
        <td>0.5639</td>
        <td>9.349</td>
    </tr>
    <tr>
        <td>microsoft_Phi-3-mini-4k-instruct</td>
        <td>fp32</td>
        <td>0.6839</td>
        <td>4.1681</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128_gptq_scale</td>
        <td>0.6757</td>
        <td>4.5107</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128_scale</td>
        <td>0.6736</td>
        <td>4.4711</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128_gptq</td>
        <td>0.6513</td>
        <td>4.8365</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128</td>
        <td>0.6342</td>
        <td>5.3419</td>
    </tr>
    <tr>
        <td>mistralai_Mistral-7B-v0.1</td>
        <td>fp32</td>
        <td>0.7592</td>
        <td>3.1898</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128_scale</td>
        <td>0.7479</td>
        <td>3.3527</td>
    </tr>
    <tr>
        <td></td>
        <td>int4_sym_r100_gs128</td>
        <td>0.7421</td>
        <td>3.4932</td>
    <t/r>
</table>

#### Accuracy/Footprint trade-off

Below are the tables showing the accuracy/footprint trade-off for `meta-llama/Llama-2-7b-chat-hf` and
`microsoft/Phi-3-mini-4k-instruct` compressed with different options.

Compression ratio is defined as the ratio between the size of fp32 model and size of the compressed one.
Accuracy metrics are measured on 3 tasks [lambada openai](https://huggingface.co/datasets/EleutherAI/lambada_openai), [wikitext](https://arxiv.org/pdf/1609.07843.pdf), [WWB](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark).
The `average relative error` in the tables below is the mean of relative errors for each of four tasks with respect to
the metric value for fp32 model. All int4 models are compressed group-wise with `group_size=64` and `mode=CompressionMode.INT4_ASYM` and
with calibration dataset based on 128 samples from `wikitext-2-v1`. Int8 model is compressed with `mode=CompressionMode.INT8_ASYM`.
The following advanced parameters were used for AWQ, Scale Estimation and Lora Correction algorithms:

```python
AdvancedCompressionParameters(
  awq_params=AdvancedAWQParameters(32, 0.05, 0.0, 1.0, 100),
  scale_estimation_params=AdvancedScaleEstimationParameters(32, 5, 10, -1.0),
  lora_correction_params=AdvancedLoraCorrectionParameters(adapter_rank=<LORA_RANK>)
)
```

The tables clearly shows the followings:

- More layers in 8 bit does improve accuracy, but it also increases the footprint significantly.
- Scale Estimation, AWQ, GPTQ improve the accuracy of the baseline int4 model without increasing the footprint.
- The Lora Correction algorithm further improves the accuracy of int4 models with a much smaller footprint compared to mixed-precision models that have the same or worse accuracy.

Accuracy/footprint trade-off for `meta-llama/Llama-2-7b-chat-hf`:

| mode                                            | %int4   | %int8   | lora<br>rank   | average<br>relative<br>error   | compression<br>rate   |
|:------------------------------------------------|:--------|:--------|:---------------|:-------------------------------|:----------------------|
| fp32                                            | 0%      | 0%      |                | 0.0%                           | 1.0x                  |
| int4 + awq + scale estimation + lora correction | 100%    | 0%      | 256.0          | 2.5%                           | 6.1x                  |
| int4 + awq + scale estimation                   | 40%     | 60%     |                | 2.5%                           | 4.8x                  |
| int4 + awq + scale estimation                   | 60%     | 40%     |                | 2.7%                           | 5.4x                  |
| int4 + awq + scale estimation                   | 80%     | 20%     |                | 3.5%                           | 6.2x                  |
| int4 + awq + scale estimation + lora correction | 100%    | 0%      | 128.0          | 3.6%                           | 6.6x                  |
| int4 + awq + scale estimation + lora correction | 100%    | 0%      | 32.0           | 3.9%                           | 7.0x                  |
| int4 + awq + scale estimation + gptq            | 100%    | 0%      |                | 4.1%                           | 7.2x                  |
| int4 + awq + scale estimation                   | 100%    | 0%      |                | 5.3%                           | 7.2x                  |
| int4                                            | 100%    | 0%      |                | 8.5%                           | 7.2x                  |

![alt text](llama2_asym.png)

Accuracy/footprint trade-off for `microsoft/Phi-3-mini-4k-instruct`:

| mode                                      | %int4   | %int8   | lora<br>rank   | average<br>relative<br>error   | compression<br>rate   |
|:------------------------------------------|:--------|:--------|:---------------|:-------------------------------|:----------------------|
| fp32                                      | 0%      | 0%      |                | 0.0%                           | 1.0x                  |
| int8                                      | 0%      | 100%    |                | 1.0%                           | 4.0x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 256.0          | 3.9%                           | 6.0x                  |
| int4 + scale estimation                   | 40%     | 60%     |                | 4.1%                           | 4.8x                  |
| int4 + scale estimation                   | 60%     | 40%     |                | 4.3%                           | 5.4x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 128.0          | 4.6%                           | 6.5x                  |
| int4 + scale estimation                   | 80%     | 20%     |                | 5.7%                           | 6.1x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 8.0            | 5.8%                           | 7.1x                  |
| int4 + scale estimation + gptq            | 100%    | 0%      |                | 6.1%                           | 7.1x                  |
| int4 + scale estimation                   | 100%    | 0%      |                | 7.5%                           | 7.1x                  |
| int4                                      | 100%    | 0%      |                | 11.9%                          | 7.1x                  |

![alt text](phi3_asym.png)

### Limitations

- The algorithm is supported for OpenVINO, PyTorch and Torch FX models.
- The compression applies in-place.
- The compressed model is not trainable.
- INT4_SYM, INT4_ASYM, NF4 and E2M1 modes, grouped quantization and mixed precision selection is available for OpenVINO backend only.
- NF4, E2M1 support is experimental on GPU and NPU - models quantized to nf4/e2m1 should not be faster models quantized to 8-bit integer.

### Additional resources

- [LLM Weight Compression](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html)
- [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
- [Inference with Hugging Face and Optimum Intel](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html)
- [Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference)
- [Large Language Models Weight Compression Example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama)
- [Tuning Ratio and Group Size Example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams)

List of notebooks demonstrating OpenVINO conversion and inference together with NNCF weight compression for models from various domains:

- [LLM Instruction Following](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-question-answering)
- [LLM Chat Bots](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot)
