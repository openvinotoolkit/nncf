# GPTQModel convertor

> [!WARNING]
> It's experimental feature with limitations. The API and functionality may change in future releases.

The module to convert compressed model by NNCF to [GPTQModel](https://github.com/ModelCloud/GPTQModel) format.

## How it works

The convertor replaces compressed linear layers with GPTQModel layers, preserving the quantization parameters and weights.
This allows for efficient inference using GPTQModel's optimized implementations.

## Usage

```python
from nncf.experimental.torch.gptqmodel.convertor import convert_model

nncf_model = nncf.compress_weights(
    model,
    dataset=nncf.Dataset([example_input]),
    mode=nncf.CompressWeightsMode.INT8_ASYM,
)
converted_model = convert_model(nncf_model)
```

## Current limitations

- Support only INT8 compression type
- Should compress only linear operation inside nn.Linear module
- Can not combine execution NNCF hooks and GPTQModel layers
- Not support save and load
- Only TritonV2QuantLinear support
