# Uniform Quantization with Fine-Tuning

>_Scroll down for the examples of the JSON configuration files that can be used to apply this algorithm_.

A uniform "fake" quantization method supports an arbitrary number of bits (>=2) which is used to represent weights and activations.
The method performs differentiable sampling of the continuous signal (for example, activations or weights) during forward pass, simulating inference with integer arithmetic.

## Common Quantization Formula

Quantization is parametrized by clamping range and number of quantization levels. The sampling formula is the following:

$ZP = \lfloor - input\\_low * s \rceil$

$output = \frac{\left\lfloor (clamp(input; input\\_low, input\\_high)-input\\_low) * s- ZP \right\rceil} {s}$

$clamp(input; input\\_low, input\\_high)$

$s = \frac{levels - 1}{input\\_high - input\\_low}$

$input\\_low$ and $input\\_high$ represent the quantization range and $\left\lfloor \cdot \right\rceil$ denotes rounding to the nearest integer.

## Symmetric Quantization

During the training, we optimize the **scale** parameter that represents the range `[input_low, input_range]` of the original signal using gradient descent:

$input\\_low=scale*\frac{level\\_low}{level\\_high}$

$input\\_high=scale$

In the formula above, $level\\_low$ and $level\\_high$ represent the range of the discrete signal.

- For weights:

    $level\\_low=-2^{bits-1}+1$

    $level\\_high=2^{bits-1}-1$

    $levels=255$

- For unsigned activations:

    $level\\_low=0$

    $level\\_high=2^{bits}-1$

    $levels=256$

- For signed activations:

    $level\\_low=-2^{bits-1}$

    $level\\_high=2^{bits-1}-1$

    $levels=256$

For all the cases listed above, the common quantization formula is simplified after substitution of $input\\_low$, $input\\_high$ and $levels$:

$output = \left\lfloor clamp(input * \frac{level\\_high}{scale}, level\\_low, level\\_high)\right \rceil * \frac{scale}{level\\_high}$

Use the `num_init_samples` parameter from the `initializer` group to initialize the values of `scale` and determine which activation should be signed or unsigned from the collected statistics using given number of samples.

## Asymmetric Quantization

During the training we optimize the `input_low` and `input_range` parameters using gradient descent:

$input\\_high=input\\_low + input\\_range$

$levels=256$

$level\\_low=0$

$level\\_high=2^{bits}-1$

For better accuracy, floating-point zero should be within quantization range and strictly mapped into quant (without rounding). Therefore, the following scheme is applied to ranges of weight and activation quantizers before applying actual quantization:

${input\\_low}' = min(input\\_low, 0)$

${input\\_high}' = max(input\\_high, 0)$

$ZP= \left\lfloor \frac{-{input\\_low}'*(levels-1)}{{input\\_high}'-{input\\_low}'} \right \rceil$

${input\\_high}''=\frac{ZP-levels+1}{ZP}*{input\\_low}'$

${input\\_low}''=\frac{ZP}{ZP-levels+1}*{input\\_high}'$

$$
\begin{flalign} &
{input\\_low,input\\_high} = \begin{cases} {input\\_low}',{input\\_high}', \& ZP \in {0,levels-1} \\
{input\\_low}',{input\\_high}'', \& {input\\_high}'' - {input\\_low}' > {input\\_high}' - {input\\_low}'' \\
{input\\_low}'',{input\\_high}', \& {input\\_high}'' - {input\\_low}' <= {input\\_high}' - {input\\_low}''
\end{cases}
&\end{flalign}
$$

You can use the `num_init_samples` parameter from the `initializer` group to initialize the values of `input_low` and `input_range` from the collected statistics using given number of samples.

## Quantizer setup and hardware config files

NNCF allows to quantize models for best results on a given Intel hardware type when executed using OpenVINO runtime.
To achieve this, the quantizer setup should be performed with following considerations in mind:

1. every operation that can accept quantized inputs on a given HW (i.e. can be executed using quantized input values) should have its inputs quantized in NNCF
2. the quantized inputs should be quantized with a configuration that is supported on a given HW for a given operation (e.g. per-tensor vs per-channel quantization, or 8 bits vs. 4 bits)
3. for operations that are agnostic to quantization, the execution should handle quantized tensors rather than full-precision tensors.
4. certain operation sequences will be runtime-optimized to execute in a single kernel call ("fused"), and additional quantizer insertion/quantization simulation within such operation sequences will be detrimental to overall performance

These requirements are fulfilled by the quantizer propagation algorithm.
The algorithm first searches the internal NNCF representation of the model's control flow graph for predefined patterns that are "fusible", and apply the fusing to the internal graph representation as well.
Next, the operations in the graph that can be associated to input-quantizable operations on a given target hardware are assigned a single quantizer for each its quantizable activation input, with a number of possible quantizer configurations attached to it (that are feasible on target HW).
The quantizers are then "propagated" against the data flow in the model's control flow graph as far as possible, potentially merging with other quantizers.
Once all quantizers have reached a standstill in their propagation process, each will have a final (possibly reduced) set of possible quantizer configurations, from which a single one is either chosen manually, or using a precision initialization algorithm (which accepts the potential quantizer locations and associated potential quantizer configuration sets).
The resulting configuration is then applied as a final quantizer setup configuration.

Note that this algorithm applies to activation quantization only - the weight quantizers do not require propagation.
However, the possible configurations of weight quantizers themselves are also sourced from the HW config file definitions.

The HW to target for a given quantization algorithm run can be specified in NNCF config using the global `"target_device"` option.
The default corresponds to CPU-friendly quantization.
`"TRIAL"` corresponds to a configuration that uses the general quantizer propagation algorithm, but does not use any HW-specific information about quantizability of given operation types or possible quantizer configs for associated inputs or operation weights.
Instead it uses a default, basic 8-bit symmetric per-tensor quantization configuration for each quantizer, and quantizes inputs of a certain default operation set, which at the moment is defined internally in NNCF.
The quantization configuration in the `"target_device": "TRIAL"` case may be overridden using the regular `"activations"` and `"weights"` sections in the quantization compression algorithm sub-config, see below.

For all target HW types, parts of the model graph can be marked as non-quantizable by using the `"ignored_scopes"` field - inputs and weights of matching nodes in the NNCF internal graph representation will not be quantized, and the downstream quantizers will not propagate upwards through such nodes.

## Quantization Implementation

In our implementation, we use a slightly transformed formula. It is equivalent by order of floating-point operations to simplified symmetric formula and the asymmetric one. The small difference is addition of small positive number `eps` to prevent division by zero and taking absolute value of range, since it might become negative on backward:

$output = \frac{clamp(\left\lfloor (input-input\\_low^{*}) *s - ZP \right \rceil, level\\_low, level\\_high)}{s}$

$s = \frac{level\\_high}{|input\\_range^{*}| + eps}$

$ZP = \lfloor-input\\_low * s\rceil$

For asymmetric:

$input\\_low^{*} = input\\_low$

$input\\_range^{*} = input\\_range$

For symmetric:

$input\\_low^{*} = 0$

$input\\_range^{*} = scale$

The most common case of applying quantization is 8-bit uniform quantization.
NNCF example scripts provide a plethora of configuration files that implement this case ([PyTorch](/examples/torch/classification/configs/quantization/inception_v3_imagenet_int8.json), [TensorFlow](/examples/tensorflow/classification/configs/quantization/inception_v3_imagenet_int8.json))

---

**NOTE**

There is a known issue with AVX2 and AVX512 CPU devices. The issue appears with 8-bit matrix calculations with tensors which elements are close to the maximum or saturated.
AVX2 and AVX512 utilize a 16-bit register to store the result of operations on tensors. In case when tensors are saturated the buffer overflow happens.
This leads to accuracy degradation. For more details of the overflow issue please refer [here](https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html).

To fix this issue inside NNCF, by default, all weight tensors are quantized in 8 bits but only 7 bits are effectively used.
This regime is used when `"target_device": "CPU"` or `"target_device": "ANY"` set. This fix, potentially, requires longer fine-tuning.

To control the application of overflow fix, `"overflow_fix"` config option is introduced. The default value is `"overflow_fix": "enable"`. To apply the overflow issue fix only to the first layer, use `"overflow_fix": "first_layer_only"`. To disable the overflow issue fix for all layers, use `"overflow_fix": "disable"`.

---

<a name="mixed_precision_quantization"></a>

## Mixed-Precision Quantization

Quantization to lower precisions (e.g. 6, 4, 2 bits) is an efficient way to accelerate inference of neural networks.
Although NNCF supports quantization with an arbitrary number of bits to represent weights and activations values,
choosing ultra-low bitwidth could noticeably affect the model's accuracy. A good trade-off between accuracy and performance is achieved by assigning different precisions to different layers. NNCF provides two automatic precision assignment algorithms, namely **HAWQ** and **AutoQ**.

### HAWQ

NNCF utilizes the [HAWQ-v2](https://arxiv.org/pdf/1911.03852.pdf) method to automatically choose optimal mixed-precision
configuration by taking into account the sensitivity of each layer, i.e. how much lower-bit quantization of each layer
decreases the accuracy of model. The most sensitive layers are kept at higher precision. The sensitivity of the i-th layer is
calculated by multiplying the average Hessian trace with the L2 norm of quantization perturbation:

$\overline{Tr}(H_{i}) * \left \|\| Q(W_{i}) - W_{i} \right \|\|^2_2$

The sum of the sensitivities for each layer forms a metric which serves as a proxy to the accuracy of the compressed
model: the lower the metric, the more accurate should be the corresponding mixed precision model on the validation
dataset.

To find the optimal trade-off between accuracy and performance of the mixed precision model we also compute a
compression ratio - the ratio between **bit complexity** of a fully INT8 model and mixed-precision lower bitwidth one.
The bit complexity of the model is a sum of bit complexities for each quantized layer, which are defined as a product
of the layer FLOPS and the quantization bitwidth. The optimal configuration is found by calculating the sensitivity
metric and the compression ratio for all possible bitwidth settings and selecting the one with the minimal metric value
among all configurations with a compression ratio below the specified threshold.

By default, the compression ratio is 1.5. It should be enough to compress the model with no more than 1% accuracy drop.
But if it doesn't happen, the lower ratio can be set by `compression_ratio` parameter in the `precision` section of
configuration file. E.g. uniformly int8 quantized model is 1 in compression ratio, 2 - for uniform int4 quantization, 0.25 - for uniform int32 quantization.

To avoid the exponential search procedure, we apply the following restriction: layers with a small average Hessian
trace value are quantized to lower bitwidth and vice versa.

The Hessian trace is estimated with the randomized [Hutchinson algorithm](https://www.researchgate.net/publication/220432178_Randomized_Algorithms_for_Estimating_the_Trace_of_an_Implicit_Symmetric_Positive_Semi-Definite_Matrix).
Given Rademacher distributed random vector v, the trace of symmetric matrix H is equal to the estimation of a quadratic form:

$Tr(H) = \mathbb{E}[v^T H v]$

The randomized algorithm solves the expectation by Monte Carlo using sampling of v from its distribution, evaluating
the quadratic term, and averaging:

$Tr(H) \approx \frac{1}{m}\sum\limits_{i=1}^{m}[v_i^T H v_i]$

Evaluation of the quadratic term happens by computing ![Hv](https://latex.codecogs.com/png.latex?Hv) - the result
of multiplication of the Hessian matrix with a given random vector v, without the explicit formation of the Hessian operator.
For gradient of the loss with respect to the i-th block ![g_i](https://latex.codecogs.com/png.latex?g_i) and for
a random vector v, which is independent of ![W_i](https://latex.codecogs.com/png.latex?W_i), we have the equation:

$\frac{\partial(g_i^T v)}{\partial  W_i} = H_i v$

where $H_i$ is the Hessian matrix of loss with respect to
$W_i$. Hence $Hv$ can be
computed by 2 backpropagation passes: first  - with respect to the loss and second - with respect to the product of the
gradients and a random vector.

The aforementioned procedure sets bitwidth for weight quantizers only. Bitwidth for activation quantizers is assigned
on the next step in two ways: strict or liberal. All quantizers between modules with quantizable inputs have the same
bitwidth in the strict mode. Liberal mode allows different precisions within the group. For both cases, bitwidth is
assigned based on the rules of the hardware config. If multiple variants are possible the minimal compatible bitwidth
is chosen. By default, liberal mode is used as it does not reject a large number of possible bitwidth settings.
The `bitwidth_assignment_mode` parameter can override it to the strict one.

For automatic mixed-precision selection it's recommended to use the following template of configuration file:

```json
    "optimizer": {
        "base_lr": 3.1e-4,
        "schedule_type": "plateau",
        "type": "Adam",
        "schedule_params": {
            "threshold": 0.1,
            "cooldown": 3
        },
        "weight_decay": 1e-05
    },
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "precision": {
                "type": "hawq",
                "bits": [4,8]
                "compression_ratio": 1.5
            }
        }
    }
```

Note, optimizer parameters are model specific, this template contains optimal ones for ResNet-like models.

The [example](/examples/torch/classification/configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq.json) of
using the template in a full-fledged configuration file is provided with the [classification sample](/examples/torch/classification/README.md) for PyTorch.

This template uses `plateau` scheduler. Though it usually leads to a lot of epochs of tuning for achieving a good
model's accuracy, this is the most reliable way. Staged quantization is an alternative approach and can be more than
two times faster, but it may require tweaking of hyper-parameters for each model. Please refer to configuration files
ending by `*_staged` for an example of this method.

The manual mode of mixed-precision quantization is also available by explicitly setting the bitwidth per layer
 through `bitwidth_per_scope` parameter.

---
**NOTE**

Precision initialization overrides bits settings specified in `weights` and `activations` sections of configuration
file.

---

### AutoQ

NNCF provides an alternate mode, namely AutoQ, for mixed-precision automation. It is an AutoML-based technique that automatically learns the layer-wise bitwidth with explored experiences. Based on [HAQ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf), AutoQ utilizes an actor-critic algorithm, Deep Deterministic Policy Gradient (DDPG) for efficient search over the bitwidth space. DDPG is trained in an episodic fashion, converging to a deterministic mixed-precision policy after a number of episodes. An episode is constituted by stepping, the DDPG transitions from quantizer to quantizer sequentially to predict a precision of a layer. Each quantizer essentially denotes a state in RL framework and it is represented by attributes of the associated layers. For example, a quantizer for 2D Convolution is represented by its quantizer Id (integer), input and output channel size, feature map dimension, stride size, if it is depthwise, number of parameters etc. It is recommended to check out ```_get_layer_attr``` in [```quantization_env.py```](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/automl/environment/quantization_env.py#L333) for the featurization of different network layer types.

When the agent enters a state/quantizer, it receives the state features and forward passes them through its network. The output of the forward pass is a scalar continuous action output which is subsequently mapped to the bitwidth options of the particular quantizer. The episode terminates after the prediction of the last quantizer and a complete layer-wise mixed-precision policy is obtained. To ensure a policy fits in the user-specified compression ratio, the policy is post processed by reducing the precision sequentially from the last quantizer until the compression ratio is met.

To evaluate the goodness of a policy, NNCF backend quantizes the workload accordingly and performs evaluation with the user-registered function. The evaluated score, together with the state embedding, predicted action are appended to an experience vault to serve for DDPG learning. The learning is carried out by sampling the data point from the experience vault for supervised training of the DDPG network. This process typically happens at a fixed interval. In the current implementation, it is performed after each episode evaluation. For bootstrapping, exploration and diversity of experience, noise is added to action output. As the episodic iterations progress, the noise magnitude is gradually reduced to zero, a deterministic mixed-precision policy is converged at the end of the episodes. NNCF currently keeps track of the best policy and uses it for fine tuning.

```json5
{
   "target_device": "NPU",
   "compression": {
      "algorithm": "quantization",
      "initializer": {
         "precision": {
            "type": "autoq",
            "bits": [
               2,
               4,
               8
            ],
            "iter_number": 300,
            "compression_ratio": 0.15,
            "eval_subset_ratio": 0.20,
            "dump_init_precision_data": true
         }
      }
   }
}
```

The snippet above demonstrates the specification of AutoQ in NNCF config. ```target_device``` determines the bitwidth choices available for a particular layer. ```bits``` also defines the precision space of quantizer but it is only active in the absence of target device.

```iter_number``` is synonymous to the number of episodes. A good choice depends on the number of quantizers in a workload and also the number of bitwidth choice. The larger the number, more episodes are required.

```compression_ratio``` is the target model size after quantization, relative to total parameters size in FP32. E.g. uniformly int8 quantized model is 0.25 in compression ratio, 0.125 for uniform int4 quantization.

```eval_subset_ratio``` is ratio of dataset to be used for evaluation for each iteration. It is used by the callback function. (See below).

```dump_init_precision_data``` dumps AutoQ's episodic metrics as tensorboard events, viewable in Tensorboard.

As briefly mentioned earlier, user is required to register a callback function for policy evaluation. The interface of the callback is a model object and torch loader object. The callback must return a scalar metric. The callback function and a torch loader are registered via ```register_default_init_args```.

Following is an example of wrapping ImageNet validation loop as a callback. Top5 accuracy is chosen as the scalar objective metric. ```autoq_eval_fn``` and ```val_loader``` are registered in the call of ```register_default_init_args```.

```python
    def autoq_eval_fn(model, eval_loader):
        _, top5 = validate(eval_loader, model, criterion, config)
        return top5

    nncf_config = register_default_init_args(
            nncf_config, init_loader, criterion, train_criterion_fn,
            autoq_eval_fn, val_loader, config.device)
```

The complete config [example](/examples/torch/classification/configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_autoq_staged.json) that applies AutoQ to MobileNetV2 is provided within the [classification sample](/examples/torch/classification/README.md) for PyTorch.

## Example configuration files

>_For the full list of the algorithm configuration parameters via config file, see the corresponding section in the [NNCF config schema](https://openvinotoolkit.github.io/nncf/)_.

- Quantize a model using default algorithm settings (8-bit, quantizers configuration chosen to be compatible with all Intel target HW types):

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization"
    }
}
```

- Quantize a model to 8-bit precision targeted for Intel CPUs, with additional constraints of symmetric weight quantization and asymmetric activation quantization:

```json5
{
    "input_info": { "sample_size": [1, 3, 32, 32] }, // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization",
       "weights": {"mode": "symmetric"},
       "activations": {"mode": "asymmetric"}
    },
   "target_device": "CPU"
}
```

- Quantize a model with fully symmetric INT8 quantization and increased number of quantizer range initialization samples (make sure to supply a corresponding data loader in code via `nncf.config.structures.QuantizationRangeInitArgs` or the `register_default_init_args` helper function):

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization",
       "mode": "symmetric",
       "initializer": {
         "range": { "num_init_samples": 5000 }
       }
    }
}
```

- Quantize a model using 4-bit per-channel quantization for experimentation/trial purposes (end-to-end performance and/or compatibility with OpenVINO Inference Engine not guaranteed)

```json5
{
    "input_info": { "sample_size": [1, 3, 32, 32] }, // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization",
       "bits": 4,
       "per_channel": true
    },
    "target_device": "TRIAL"
}
```

- Quantize a multi-input model to 8-bit precision targeted for Intel CPUs, with a range initialization performed using percentile statistics (empirically known to be better for NLP models, for example) and excluding some parts of the model from quantization:

```json5
{
    "input_info": [
       {
            "keyword": "input_ids",
            "sample_size": [1, 128],
            "type": "long",
            "filler": "ones"
        },
        {
            "keyword": "attention_mask",
            "sample_size": [1, 128],
            "type": "long",
            "filler": "ones"
        }
    ], // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization",
       "initializer": {
          "range": {
             "num_init_samples": 64,
             "type": "percentile",
             "params": {
                "min_percentile": 0.01,
                "max_percentile": 99.99
             }
          }
       },
       "ignored_scopes": ["{re}BertSelfAttention\\[self\\]/__add___0",
            "RobertaForSequenceClassification/RobertaClassificationHead[classifier]/Linear[out_proj]",
            "RobertaForSequenceClassification/RobertaClassificationHead[classifier]/Linear[dense]"
        ]
    },
    "target_device": "TRIAL"
}
```

- Quantize a model to variable bit width using 300 iterations of the AutoQ algorithm, with a target model size (w.r.t the effective parameter storage size) set to 15% of the FP32 model and possible quantizer bitwidths limited to INT2, INT4 or INT8.

```json5
{
    "input_info": { "sample_size": [1, 3, 224, 224] }, // the input shape of your model may vary
    "compression": {
       "algorithm": "quantization",
       "initializer": {
           "precision": {
               "type": "autoq", // or "type": "hawq"
               "bits": [2, 4, 8],
               "compression_ratio": 0.15,
               "iter_number": 300
           }
       }
    },
    "target_device": "TRIAL"
}
```
