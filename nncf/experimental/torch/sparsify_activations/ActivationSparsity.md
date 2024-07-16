### Activation Sparsity

The `sparsify_activations` algorithm is a post-training method designed to introduce sparsity into the activations of a neural network. This process reduces the number of active neurons during inference by masking out neurons based on their magnitude relative to a calibrated static threshold.

The algorithm sparsifies the input of a layer by applying the following function:

$$ 
sparsify(X) = 
\begin{cases} 
\cdot & \text{if } |\cdot| \ge \tau \\
0 & \text{if } |\cdot| < \tau 
\end{cases}
$$

The magnitude threshold $\tau$ that corresponds to a desired level of sparsity is determined by the statistical quantile value of activations collected via an input dataset:

$$
\tau = Quantile(|X|,\ target\ sparsity)
$$

`sparsify_activations` automates the process of identifying the pruning thresholds based on user-specified layers, target sparsities and input dataset.

#### Example Usage

Below is an example of applying `sparsify_activations` algorithm to a torch model. Optionally, you can also call `nncf.compress_weights()` before sparsification to get an optimized model with quantized weights and sparse activations.

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

In this example, we first conduct data-free INT8 asymmetric weight quantization on the model. Then we do activation sparsification, setting the target activation sparsity to 30% for all the layers containing the keywords "up_proj" and "gate_proj", and 50% for layers with "down_proj" keyword.

#### Interface Details

- `model`: The model to be sparsified. Currently only Torch backend is supported.
- `dataset`: A dataset to calibrate the pruning thresholds. **TODO** NNCF Dataset
- `target_sparsity_by_scope`: A dictionary defines the target activation sparsity level for specified layers. For each item, the key is an instance of `TargetScope` class representing the layers to match in the model's NNCF graph; the corresponding value is a float number in the range [0, 1] representing the target sparsity level. `TargetScope` supports absolute and REGEX-based name matching.

  - Example:

    ```python
    {
        # Target sparsity is 60% for node "Dummy/Linear[layer]/linear_0" in the model graph
        TargetScope(names=["Dummy/Linear[layer]/linear_0"]): 0.6,
        # Target sparsity is 30% for the layers whose name contains "up_proj" or "down_proj".
        TargetScope(patterns=[".*up_proj.*", ".*down_proj.*"]): 0.3,
    }
    ```

- `ignored_scope`: Optional. It defines the nodes in the model graph that should be ignored by this algorithm. Note that unsupported layer types are already filtered out internally, so there is no need to mention them in `ignored_scope`. The algorithm currently only supports Linear layer.


#### Evaluation results
> TODO

#### Known Limitations

1. When used with `nncf.compress_weight`,  only int8 is supported. can it work before or after? **TODO**
2. Actual activation sparsity during inference is dynamic and per input basis, deviation from the target should be expected. In our local experiments, the statistical mean of actual activation sparsity aligned to the target when thresholds are calibrated on datasets similar to the final task.
3. Similar to other compression methods, model accuracy and activation sparsity are trade-off at play. For large language models like [Llama](https://llama.meta.com), it is recommended to start with 30%~50% sparsity for the linear layers in feed-forward networks.
