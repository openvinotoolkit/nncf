### Activation Sparsity (experimental feature)

The `sparsify_activations` algorithm is a post-training method designed to introduce sparsity into the activations of a neural network. This process reduces the number of active neurons during inference by masking out neurons based on their magnitude relative to a calibrated static threshold.

The algorithm sparsifies the input of a layer by applying the following function:

$$
sparsify(X) =
\begin{cases}
X & \text{if } |X| > \tau \\
0 & \text{if } |X| \le \tau
\end{cases}
$$

The magnitude threshold $\tau$ that corresponds to a desired level of sparsity is determined by the statistical quantile value of activations collected via an input dataset:

$$
\tau = Quantile(|X|,\ target\ sparsity)
$$

`sparsify_activations` automates the process of identifying the pruning thresholds based on user-specified layers, target sparsities and input dataset.

> Note: This feature is **experimental** and intended solely for evaluation of sparsity-task performance. While activation sparsity can improve inference efficiency of decoding phase for Large Language Models (LLMs) ([Liu et al., 2023](https://arxiv.org/abs/2310.17157)), it neccessitates optimized runtime kernels, which are in development.

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
    }
)
```

In this example, we first conduct data-free INT8 asymmetric weight quantization on the model. Then we do activation sparsification, setting the target activation sparsity to 30% for all the layers containing the keywords "up_proj" and "gate_proj", and 50% for layers with "down_proj" keyword.

#### Interface Details

- `model`: The model to be sparsified. Currently only Torch backend is supported.
- `dataset`: An `nncf.Dataset` instance used to calibrate the pruning thresholds.
- `target_sparsity_by_scope`: A dictionary that defines the target activation sparsity level for specified layers. For each item, the key is an instance of `TargetScope` class representing the layers to match in the model's NNCF graph; the corresponding value is a float number in the range [0, 1] representing the target sparsity level. `TargetScope` supports absolute and REGEX-based name matching.

  - Example:

    ```python
    {
        # Target sparsity is 60% for node "Dummy/Linear[layer]/linear_0" in the model graph
        TargetScope(names=["Dummy/Linear[layer]/linear_0"]): 0.6,
        # Target sparsity is 30% for the layers whose name contains "up_proj" or "down_proj".
        TargetScope(patterns=[".*up_proj.*", ".*down_proj.*"]): 0.3,
    }
    ```

- `ignored_scope`: Optional. If specified, it should be an instance of `nncf.IgnoredScope` class that defines the nodes in the model graph to be ignored by this algorithm. Note that unsupported layer types are already filtered out internally, so there is no need to mention them in `ignored_scope`. The algorithm currently only supports Linear layers, as they benefit most from dynamic sparse activations by reducing memory read bandwidth for the large Linear layer weights used in LLMs.

#### Evaluation results

Here is the word perplexity for different language models on a subset of [wikitext dataset](https://arxiv.org/abs/1609.07843), with maximum context length set as 2048. In the table, "int8_asym" means the model weights are asymmetrically quantized to INT8. "up/gate/down" means the up, gate, and down projection layers in the [Gated Linear Units](https://arxiv.org/abs/1612.08083) (GLU) style feed forward networks. "Avg. Activation Sparsity" column shows the average activation sparsity on the evaluation samples. For example, "down50%" means that on average the input activations of all "down" layers have a sparsity of 50%.

<table>
    <tr bgcolor='#B4B5BB'>
        <td>Model</td>
        <td>Mode</td>
        <td>Avg. Activation Sparsity</td>
        <td>Word Perplexity (↓)</td>
    </tr>
        <tr>
        <td>meta-llama/Llama-2-7b-hf</td>
        <td>fp32</td>
        <td>-</td>
        <td>9.242</td>
    </tr>
        <tr>
        <td></td>
        <td>sparse_activation</td>
        <td>up/gate30% + down50%</td>
        <td>9.508</td>
    </tr>
        <tr>
        <td></td>
        <td>int8_asym + sparse_activation</td>
         <td>up/gate30% + down50%</td>
        <td>9.511</td>
    </tr>
        <tr>
        <td>meta-llama/Meta-Llama-3-8B-Instruct</td>
        <td>fp32</td>
        <td>-</td>
        <td>10.802</td>
    </tr>
        <tr>
        <td></td>
        <td>sparse_activation</td>
        <td>up/gate30% + down50%</td>
        <td>11.294</td>
    </tr>
        <tr>
        <td></td>
        <td>int8_asym + sparse_activation</td>
         <td>up/gate30% + down50%</td>
        <td>11.302</td>
    </tr>
        <tr>
        <td>mistralai/Mixtral-8x7B-Instruct-v0.1</td>
        <td>fp32</td>
        <td>-</td>
        <td>6.224</td>
    </tr>
        <tr>
        <td></td>
        <td>sparse_activation</td>
        <td>up/gate40% + down50%</td>
        <td>6.561</td>
    </tr>
        <tr>
        <td></td>
        <td>int8_asym + sparse_activation</td>
         <td>up/gate40% + down50%</td>
        <td>6.579</td>
    </tr>
</table>

#### Known Limitations

1. Currently activation sparsity only supports Torch backend. Consequently, this restricts the available compression modes to 8-bit integer modes when using `nncf.compress_weights()` before activation sparsification. More information on supported modes can be found at [Weights Compression](../../../../docs/usage/post_training_compression/weights_compression/Usage.md#limitations).
2. Actual activation sparsity during inference is dynamic and per input basis, deviation from the target should be expected. In our local experiments, the statistical mean of actual activation sparsity aligned to the target when thresholds are calibrated on datasets similar to the final task.
3. Similar to other compression methods, model accuracy and activation sparsity are trade-off at play. For LLMs like [Llama](https://llama.meta.com), it is recommended to start with 30%~50% sparsity for the Linear layers in feed-forward networks.
