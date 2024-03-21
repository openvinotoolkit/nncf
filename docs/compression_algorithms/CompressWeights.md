### Weights Compression

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run Weights Compression with, and PyTorch is also supported.

#### The algorithm description

The Weights Compression algorithm is aimed at compressing the weights of the models and can be used to optimize the model footprint and performance of large models where the size of weights is relatively larger than the size of activations, for example, Large Language Models (LLM). The algorithm compresses weights for Linear, Convolution and Embedding layers.

#### Supported modes

By default, weights are compressed asymmetrically to 8-bit integer data type - "INT8_ASYM" mode.
OpenVINO backend also supports 3 modes of mixed precision weight quantization with a 4-bit data type as a primary precision - INT4_SYM, INT4_ASYM and NF4. The primary precision in case of INT4_SYM mode is unsigned 4-bit integer and weights are quantized to it [symmetrically](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#symmetric-quantization) with a fixed zero point equals to 8. In case of INT4_ASYM mode - also unsigned 4-bit integer, but weight are quantized to it [asymmetrically](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md#asymmetric-quantization) with a typical non-fixed zero point. In case of NF4 mode - [nf4](https://arxiv.org/pdf/2305.14314v1.pdf) data type without zero point.
All 4-bit modes have a grouped quantization support, when small group of weights (e.g. 128) in the channel dimension share quantization parameters (scale).
All embeddings, convolutions and last linear layers are always compressed to 8-bit integer data type. To quantize embeddings and last linear layers to 4-bit, use `all_layers=True`.
Percent of the rest layers compressed to 4-bit can be configured by "ratio" parameter. E.g. ratio=0.9 means 90% of layers compressed to the corresponding 4-bit data type and the rest to 8-bit asymmetric integer data type.

#### User guide

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

- Generally, `INT4_SYM` mode is the fastest mixed-precision mode, but it may lead to a significant accuracy degradation or perplexity increase.
  Compressing weights asymmetrically (`INT4_ASYM` mode) is the way to increase accuracy, however in turns it slows down inference a bit.
  If the accuracy or perplexity is still not satisfying, there are 2 more hyper-parameters to tune: `group_size` and `ratio`. Please refer to the [example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams) how to automatically tune these parameters.
  Lower group size and less ratio of 4-bit layers usually improve accuracy at the sacrifice of inference speed.
  Below is the example how to compress weights of 90% of layers to 4-bit integer asymmetrically with the group size 64, and
  the rest of layers to 8-bit asymmetric integer data type. The same parametrization is applicable for `INT4_SYM` mode.

```python
from nncf import compress_weights, CompressWeightsMode
compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, group_size=64, ratio=0.9) # model is openvino.Model object
```

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

- Accuracy of the 4-bit compressed models also can be improved by using AWQ algorithm over data-based mixed-precision algorithm. It is capable to equalize some subset of weights to minimize difference between
original precision and 4-bit.
Below is the example how to compress 80% of layers to 4-bit integer with a default data-based mixed precision algorithm and AWQ.
It requires to set `awq` to `True` additionally to data-based mixed-precision algorithm.

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

model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM, ratio=0.8, dataset=nncf_dataset, awq=True)

model.save_pretrained(...)
```

- `NF4` mode can be considered for improving accuracy, but currently models quantized to nf4 should not be faster models
  quantized to 8-bit asymmetric integer. Here's the example how to compress weights to nf4 data type with group size = 128.
  Different `group_size` and `ratio` are also supported.

```python
from nncf import compress_weights, CompressWeightsMode
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

#### Limitations

- The algorithm is supported for OpenVINO and PyTorch models.
- The compression applies in-place.
- The compressed model is not trainable.
- INT8_SYM, INT4_SYM, INT4_ASYM and NF4 modes, grouped quantization and mixed precision selection is available for OpenVINO backend only.
- NF4 support is experimental - models quantized to nf4 should not be faster models quantized to 8-bit integer.

#### Additional resources

- [LLM Weight Compression](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html)
- [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
- [Inference with Hugging Face and Optimum Intel](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-hf.html)
- [Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/inference)
- [Large Language Models Weight Compression Example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama)
- [Tuning Ratio and Group Size Example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams)

List of notebooks demonstrating OpenVINO conversion and inference together with NNCF weight compression for models from various domains:

- [LLM Instruction Following](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/275-llm-question-answering)
- [Dolly 2.0](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/240-dolly-2-instruction-following)
- [Stable-Zephyr-3b](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/273-stable-zephyr-3b-chatbot)
- [LLM Chat Bots](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot)
