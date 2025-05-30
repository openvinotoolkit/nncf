# Weight-Only Quantization-Aware Training (QAT) with absorbable Low-Rank Adapters (LoRA)

Use NNCF to achieve accurate LLMs with weights compressed to 4-bit via Weight-Only Quantization-Aware Training with absorbable LoRA Adapters.

<p align="center">
  <img src="/docs/usage/training_time_compression/quantization_aware_training_lora/workflow.png" alt="FQLoRA workflow" width="1000"/>
</p>

## Key Highlights

- Combines post-training compression & efficient LoRA fine-tuning.
- Achieve lossless fusion of LoRA adapters into INT4 weights with zero inference overhead.
- Introduces Neural Low-Rank Adapter Search (NLS) for improved performance on downstream tasks.
- Greatly mitigates accuracy degradation associated with quantization.

## When to apply

- When [training-free int4 weight compression](../../post_training_compression/weights_compression/Usage.md) significantly degrade accuracy
- More HW resources are available for performing efficient tuning on GPU.

## Central idea

<p align="center">
  <img src="/docs/usage/training_time_compression/quantization_aware_training_lora/absorb_lora.png" alt="absorb lora to 4bit weights" width="400"/>
</p>

The central idea - absorbable LoRA adapters. The method introduces two low-rank matrices, $A$ and $B$,
to facilitate efficient adaptation of the weight matrix $W$. This adaptation is expressed as $W' = W + B \times A$,
where $W'$ represents the modified weight matrix. Subsequently, $W'$ is processed through a FakeQuantize (FQ) operation,
denoted as $FQ(W') = \mathrm{dequantize}(\mathrm{quantize}(W'))$ to learn quantization scales for int4 compression.
In this way, LoRA adapters can be seamlessly absorbed into the 4-bit weights without accuracy loss with
zero inference overhead for adapters.

## Examples

NNCF provides two specialized FakeQuantize with LoRA adapter formats: `FQ_LORA` and `FQ_LORA_NLS`, each designed for specific use cases:

1. [Knowledge Distillation with FQ_LORA](../../../../examples/llm_compression/torch/distillation_qat_with_lora/README.md)

    <p align="center">
      <img src="/examples/llm_compression/torch/distillation_qat_with_lora/pics/training_pipeline.png" alt="distillation" width="600"/>
    </p>

    This format implements traditional low-rank adaptation with fixed ranks across all layers. It's tailored for knowledge distillation where an uncompressed model teaches a compressed student model. The [distillation_qat_with_lora](../../../../examples/llm_compression/torch/distillation_qat_with_lora/README.md) example shows it reduces accuracy loss by approximately 50% compared to NNCF post-training compression methods, without over-fitting to specific tasks.

2. [Downstream Fine-tuning with FQ_LORA_NLS](../../../../examples/llm_compression/torch/downstream_qat_with_nls/README.md)

    <p align="center">
      <img src="/examples/llm_compression/torch/downstream_qat_with_nls/pics/lora_vs_nls.png" alt="LoRA vs NLS" width="400"/>
    </p>

    This format leverages Neural Low-Rank Adapter Search (NLS) proposed by [Shears](https://arxiv.org/abs/2404.10934) and [SQFT](https://arxiv.org/abs/2410.03750), allowing flexible rank values automatically determined per layer. It eliminates manual rank hyperparameter tuning while outperforming fixed-rank LoRA on specific tasks. The [downstream_qat_with_nls](../../../../examples/llm_compression/torch/downstream_qat_with_nls/README.md) example demonstrates accuracy comparable to full-precision LoRA fine-tuning.

## Workflow

### Step 1: Apply Post Training Weight Compression

Use [Post Training Weight Compression](../../post_training_compression/weights_compression/Usage.md) to initialize model weights in either `FQ_LORA` or `FQ_LORA_NLS` format depending on the use case, described above. This introduces a FakeQuantize layer with absorbable low-rank matrices for efficient tuning.
At this point, post-training weight compression serves as initialization - ranging from simple data-free round-to-nearest methods to more advanced approaches like AWQ + Scale Estimation.
Even in data-free compression workflows with Torch, at least one sample is still needed to construct the model graph and insert FQ.

```python
model = AutoModelForCausalLM.from_pretrained(model_id)
compressed_model = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT4_ASYM,
    group_size=64,
    compression_format=nncf.CompressionFormat.FQ_LORA,
    dataset=Dataset([model.dummy_inputs])
)
```

### Step 2: Configure Training Parameters

Enable gradient updates for LoRA adapters and quantization scales. Assigning a learning rate that is 10x lower for the quantization scales can improve accuracy. See the “set_trainable” function in the samples [distillation_qat_with_lora](../../../../examples/llm_compression/torch/distillation_qat_with_lora/main.py) and [downstream_qat_with_nls](../../../../examples/llm_compression/torch/downstream_qat_with_nls/main.py) for more details.

```python
from nncf.torch.function_hook.wrapper import get_hook_storage

hook_storage = get_hook_storage(model)
model.requires_grad_(False)
for _, module in hook_storage.named_hooks():
    if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (module.num_bits == 4):
        module.enable_gradients()
```

### Step 3: Tune the Model

Choose one of the following tuning approaches:

- **Knowledge Distillation**: Use an uncompressed model as the teacher to train the compressed model as a student. See the [distillation_qat_with_lora](../../../../examples/llm_compression/torch/distillation_qat_with_lora/README.md) implementation. Compatible with `FQ_LORA` format only.

- **Downstream Task Fine-tuning**: Train directly on your target task following the [downstream_qat_with_nls](../../../../examples/llm_compression/torch/downstream_qat_with_nls/README.md) sample. Works with both `FQ_LORA` and `FQ_LORA_NLS` formats. The NLS approach delivers superior results but requires additional time to search for optimal rank configurations.

### Step 4: Convert Model to OpenVINO

To convert a PyTorch model to an INT4 OpenVINO model, transform the `FQ_LORA` or `FQ_LORA_NLS` format to the representation expected by OpenVINO (DQ format). The `nncf.strip` function handles this transformation by replacing weights with packed INT4 compressed values and adding decompression operations to the graph. Two INT4 values are packed into each INT8 value to optimize storage efficiency.

```python
# Convert to OpenVINO format after training is complete
compressed_model = nncf.strip(model, strip_format=StripFormat.DQ, example_input=example_input)
```
