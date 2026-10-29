# Large Language Models Weight Compression with Scoped Configs Example

This example demonstrates how to assign custom compression settings to specific layers of a Large Language Model (LLM) using the `scoped_configs` parameter of the NNCF weight compression API. The example applies 4/8-bit mixed-precision quantization to weights of Linear (Fully-connected) layers of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model, while explicitly controlling the precision of selected layers. This leads to a significant decrease in model footprint and performance improvement with OpenVINO.

## Scoped Configs

Each `nncf.ScopedConfig` selects layers by node `names`, regular expression `patterns`, operation `types` or `subgraphs` (the same rules as [the ignored scope](../../../../docs/usage/IgnoredScope.md)) and assigns them a compression `mode` and `group_size`:

```python
model.model = nncf.compress_weights(
    model.model,
    dataset=quantization_dataset,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    ratio=0.8,
    sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    scoped_configs=[
        nncf.ScopedConfig(
            patterns=[".*self_attn.*"],
            mode=nncf.CompressWeightsMode.INT8_ASYM,
        ),
        nncf.ScopedConfig(
            patterns=[".*mlp.down_proj.*"],
            mode=nncf.CompressWeightsMode.INT4_ASYM,
            group_size=32,
        ),
    ],
)
```

In this example:

- Attention layers (`self_attn`) are compressed to INT8 asymmetric per-channel.
- MLP down projections (`mlp.down_proj`) are compressed to INT4 asymmetric with group size 32.
- The embeddings and the last layer (`lm_head`) are compressed to the backup precision, INT8 asymmetric per-channel. By default (`all_layers=False`), they are excluded from the mixed precision selection.
- All other layers are compressed with the main settings: 80% of them (`ratio=0.8`) to INT4 symmetric, chosen by the mixed precision algorithm, and the rest to INT8.

If a layer is matched by several scoped configs, the last one in the list wins. Scoped configs also take precedence over the `ignored_scope`. See the [weight compression documentation](../../../../docs/usage/post_training_compression/weights_compression/Usage.md#custom-precision-for-specific-layers) for more details.

## Prerequisites

Before running this example, ensure you have Python 3.11+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Run Example

To run the example:

```bash
python main.py
```

This will automatically:

- Download the TinyLlama model and dataset
- Apply weight compression with scoped configs using NNCF
- Save the optimized model
- Generate an answer to a sample prompt with the compressed model
