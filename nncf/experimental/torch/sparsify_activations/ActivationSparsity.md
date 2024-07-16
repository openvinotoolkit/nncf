# Sparsify Activations

The `Sparsify Activations` algorithm introduces sparsity into the activations of a neural network, reducing the number of active neurons during inference when using optimized sparse kernels.

This algorithm sparsifies the input of a layer by zeroing out neuron \( x \) if \( \text{abs}(x) \leq \tau \), where \( \tau \) is a static threshold determined based on statistical information from a calibration dataset to meet the desired level of sparsity.

## Example Usage

Below is an example of applying Activation Sparsity algorithm to a torch model. Optionally, you can also call `nncf.compress_weights()` before sparsification to get an optimized model with quantized weights and sparse activations.

```python
import nncf
from nncf.experimental.torch.sparsify_activations import sparsify_activations, TargetScope

model = ... # Your model
dataset = ... # Calibration set

# (Optional) Weight-only quantization
model = nncf.compress_weights(
    model=model,
    mode=nncf.CompressWeightsMode.INT8_ASYM,
    dataset=dataset,
)

# Activation sparsification
model = sparsify_activations(
    model=model,
    dataset=dataset,
    target_sparsity_by_scope={
        TargetScope(patterns=[".*up_proj.*", ".*gate_proj.*"]): 0.3,
        TargetScope(patterns=[".*down_proj.*",]): 0.5,
    },
    ignored_scope=nncf.IgnoredScope(),
)
```

In this example, we first conduct data-free INT8 asymmetric weight only quantization on the model. Then we do activation sparsification, setting the target activation sparsity as 30% for all the layers named "up_proj" and "gate_proj", and 50% for layers named "down_proj".

## Interface Details

- `model`: The model to be sparsified. Currently only Torch backend is supported.
- `dataset`: A dataset to calibrate the thresholds that this algorithm uses to sparsify the neurons.
- `target_sparsity_by_scope`: Defines the target activation sparsity level for specified layers. For each item in this dict, the key is an instance of `TargetScope` class representing the layers to match in the model's NNCF graph; the corresponding value is a float number in the range [0, 1] representing the target sparsity level.

  - Example:

    ```python
    {
        # Target sparsity is 60% for node "Dummy/Linear[layer]/linear_0" in the model graph
        TargetScope(names=["Dummy/Linear[layer]/linear_0"]): 0.6,
        # Target sparsity is 30% for the layers whose name contains "up_proj" or "down_proj".
        TargetScope(patterns=[".*up_proj.*", ".*down_proj.*"]): 0.3,
    }
    ```

- `ignored_scope`: Optional. It defines the nodes in the model graph that should be ignored by this algorithm. Note that unsupported layer types are already filtered out internally, so there is no need to mention them in `ignored_scope`.

## Known Limitations

1. Activation sparsification currently supports only linear layers in the Torch backend.
2. The actual activation sparsity during inference might deviate from the target. This is because the algorithm uses static thresholds, which inevitably cannot accommodate all possible inputs. A good estimate of the thresholds may depend on the size of the calibration set, the batch size, and the quality of samples compared with the actual inference data.
3. There is a tradeoff between model accuracy and activation sparsity. For large language models like [Llama](https://llama.meta.com), it is recommended to start with 30%~50% sparsity for the linear layers in feed-forward networks.
