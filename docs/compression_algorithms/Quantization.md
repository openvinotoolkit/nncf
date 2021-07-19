
### Uniform Quantization with Fine-Tuning

A uniform "fake" quantization method supports an arbitrary number of bits (>=2) which is used to represent weights and activations.
The method performs differentiable sampling of the continuous signal (for example, activations or weights) during forward pass, simulating inference with integer arithmetic.

#### Common Quantization Formula

Quantization is parametrized by clamping range and number of quantization levels. The sampling formula is the following:

![output = \frac{\left\lfloor (clamp(input; input\_low, input\_high)-input\_low)  *s\right \rceil}{s} + input\_low\\](https://latex.codecogs.com/png.latex?output%20%3D%20%5Cfrac%7B%5Cleft%5Clfloor%20%28clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29-input%5C_low%29%20*s%5Cright%20%5Crceil%7D%7Bs%7D%20&plus;%20input%5C_low%5C%5C)

![clamp(input; input\_low, input\_high) = min(max(input, input\_low), input\_high)))](https://latex.codecogs.com/png.latex?clamp%28input%3B%20input%5C_low%2C%20input%5C_high%29%20%3D%20min%28max%28input%2C%20input%5C_low%29%2C%20input%5C_high%29%29%29)

![s=\frac{levels-1}{input\_high - input\_low}](https://latex.codecogs.com/png.latex?s%3D%5Cfrac%7Blevels-1%7D%7Binput%5C_high%20-%20input%5C_low%7D)

`input_low` and `input_high` represent the quantization range and ![\left\lfloor\cdot\right \rceil](https://latex.codecogs.com/png.latex?%5Cleft%5Clfloor%5Ccdot%5Cright%20%5Crceil) denotes rounding to the nearest integer.

####  Symmetric Quantization

During the training, we optimize the **scale** parameter that represents the range `[input_low, input_range]` of the original signal using gradient descent:

![input\_low=scale*\frac{level\_low}{level\_high}](https://latex.codecogs.com/png.latex?input%5C_low%3Dscale*%5Cfrac%7Blevel%5C_low%7D%7Blevel%5C_high%7D)

![input\_high=scale](https://latex.codecogs.com/png.latex?input%5C_high%3Dscale)

In the formula above, `level_low` and `level_high` represent the range of the discrete signal.
 - For weights:

    ![level\_low=-2^{bits-1}+1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D&plus;1),

    ![level\_high=2^{bits-1}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)

    ![levels=255](https://latex.codecogs.com/png.latex?levels%3D255)

 - For unsigned activations:

    ![level\_low=0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)

    ![level\_high=2^{bits}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)

    ![levels=256](https://latex.codecogs.com/png.latex?levels%3D256)

 - For signed activations:

    ![level\_low=-2^{bits-1}](https://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20level%5C_low%3D-2%5E%7Bbits-1%7D)

    ![level\_high=2^{bits-1}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits-1%7D-1)

    ![levels=256](https://latex.codecogs.com/png.latex?levels%3D256)

For all the cases listed above, the common quantization formula is simplified after substitution of `input_low`, `input_high` and `levels`:

![output = \left\lfloor clamp(input * \frac{level\_high}{scale}, level\_low, level\_high)\right \rceil * \frac{scale}{level\_high}](https://latex.codecogs.com/png.latex?output%20%3D%20%5Cleft%5Clfloor%20clamp%28input%20*%20%5Cfrac%7Blevel%5C_high%7D%7Bscale%7D%2C%20level%5C_low%2C%20level%5C_high%29%5Cright%20%5Crceil%20*%20%5Cfrac%7Bscale%7D%7Blevel%5C_high%7D)

Use the `num_init_samples` parameter from the `initializer` group to initialize the values of `scale` and determine which activation should be signed or unsigned from the collected statistics using given number of samples.

####  Asymmetric Quantization

During the training we optimize the **input_low** and **input_range** parameters using gradient descent:

![input\_high=input\_low + input\_range](https://latex.codecogs.com/png.latex?input%5C_high%3Dinput%5C_low%20&plus;%20input%5C_range)

![levels=256](https://latex.codecogs.com/png.latex?levels%3D256)

![level\_low=0](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_low%3D0)

![level\_high=2^{bits}-1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cdpi%7B120%7D%20level%5C_high%3D2%5E%7Bbits%7D-1)

For better accuracy, floating-point zero should be within quantization range and strictly mapped into quant (without rounding). Therefore, the following scheme is applied to ranges of weights and activations before quantization:

![{input\_low}' = min(input\_low, 0)](https://latex.codecogs.com/png.latex?%7Binput%5C_low%7D%27%20%3D%20min%28input%5C_low%2C%200%29)

![{input\_high}' = max(input\_high, 0)](https://latex.codecogs.com/png.latex?%7Binput%5C_high%7D%27%20%3D%20max%28input%5C_high%2C%200%29)

![ZP= \left\lfloor \frac{-{input\_low}'*(levels-1)}{{input\_high}'-{input\_low}'} \right \rceil ](https://latex.codecogs.com/png.latex?ZP%3D%20%5Cleft%5Clfloor%20%5Cfrac%7B-%7Binput%5C_low%7D%27*%28levels-1%29%7D%7B%7Binput%5C_high%7D%27-%7Binput%5C_low%7D%27%7D%20%5Cright%20%5Crceil)

![{input\_high}''=\frac{ZP-levels+1}{ZP}*{input\_low}'](https://latex.codecogs.com/png.latex?%7Binput%5C_high%7D%27%27%3D%5Cfrac%7BZP-levels+1%7D%7BZP%7D*%7Binput%5C_low%7D%27)

![{input\_low}''=\frac{ZP}{ZP-levels+1}*{input\_high}'](https://latex.codecogs.com/png.latex?%7Binput%5C_low%7D%27%27%3D%5Cfrac%7BZP%7D%7BZP-levels+1%7D*%7Binput%5C_high%7D%27)

![{input\_low,input\_high} = \begin{cases} {input\_low}',{input\_high}', & ZP \in $\{0,levels-1\}$ \\ {input\_low}',{input\_high}'', & {input\_high}'' - {input\_low}' > {input\_high}' - {input\_low}'' \\ {input\_low}'',{input\_high}', & {input\_high}'' - {input\_low}' <= {input\_high}' - {input\_low}''\\ \end{cases}](https://latex.codecogs.com/png.latex?%7Binput%5C_low%2Cinput%5C_high%7D%20%3D%20%5Cbegin%7Bcases%7D%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%2C%20%26%20ZP%20%5Cin%20%24%5C%7B0%2Clevels-1%5C%7D%24%20%5C%5C%20%7Binput%5C_low%7D%27%2C%7Binput%5C_high%7D%27%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3E%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%20%5C%5C%20%7Binput%5C_low%7D%27%27%2C%7Binput%5C_high%7D%27%2C%20%26%20%7Binput%5C_high%7D%27%27%20-%20%7Binput%5C_low%7D%27%20%3C%3D%20%7Binput%5C_high%7D%27%20-%20%7Binput%5C_low%7D%27%27%5C%5C%20%5Cend%7Bcases%7D)

You can use the `num_init_samples` parameter from the `initializer` group to initialize the values of `input_low` and `input_range` from the collected statistics using given number of samples.

#### Quantizer setup and hardware config files
NNCF allows to quantize models for best results on a given Intel hardware type when executed using OpenVINO runtime.
To achieve this, the quantizer setup should be performed with following considerations in mind:
1) every operation that can accept quantized inputs on a given HW (i.e. can be executed using quantized input values) should have its inputs quantized in NNCF
2) the quantized inputs should be quantized with a configuration that is supported on a given HW for a given operation (e.g. per-tensor vs per-channel quantization, or 8 bits vs. 4 bits)
3) for operations that are agnostic to quantization, the execution should handle quantized tensors rather than full-precision tensors.
4) certain operation sequences will be runtime-optimized to execute in a single kernel call ("fused"), and additional quantizer insertion/quantization simulation within such operation sequences will be detrimental to overall performance

These requirements are fulfilled by the quantizer propagation algorithm.
The algorithm first searches the internal NNCF representation of the model's control flow graph for predefined patterns that are "fusable", and apply the fusing to the internal graph representation as well.
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


#### Quantization Implementation

In our implementation, we use a slightly transformed formula. It is equivalent by order of floating-point operations to simplified symmetric formula and the assymetric one. The small difference is addition of small positive number `eps` to prevent division by zero and taking absolute value of range, since it might become negative on backward:

![output = \frac{clamp(\left\lfloor(input-input\_low^{*})*s\right\rceil, level\_low, level\_high)} {s} + input\_low^{*}](https://latex.codecogs.com/png.latex?output%3D%5Cfrac%7Bclamp%28%5Cleft%5Clfloor%28input-input%5C_low%5E%7B%2A%7D%29%2As%5Cright%5Crceil%2Clevel%5C_low%2Clevel%5C_high%29%7D%7Bs%7D%2Binput%5C_low%5E%7B%2A%7D)

![s = \frac{level\_high}{|input\_range^{*}| + eps}](https://latex.codecogs.com/png.latex?s%20%3D%20%5Cfrac%7Blevel%5C_high%7D%7B%7Cinput%5C_range%5E%7B*%7D%7C%20&plus;%20eps%7D)

For asymmetric:
![\\input\_low^{*} = input\_low \\ input\_range^{*} = input\_range ](https://latex.codecogs.com/png.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%20input%5C_low%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20input%5C_range)

For symmetric:
![\\input\_low^{*} = 0 \\ input\_range^{*} = scale](https://latex.codecogs.com/png.latex?%5C%5Cinput%5C_low%5E%7B*%7D%20%3D%200%20%5C%5C%20input%5C_range%5E%7B*%7D%20%3D%20scale)

---
**NOTE**

There is a known issue with AVX2 and AVX512 CPU devices. The issue appears with 8-bit matrix calculations with tensors which elements are close to the maximum or saturated.
AVX2 and AVX512 utilize a 16-bit register to store the result of operations on tensors. In case when tensors are saturated the buffer overflow happens.
This leads to accuracy degradation.

To fix this issue inside NNCF, weight tensors are quantized in 8 bits but only 7 bits are effectively used.
This regime is used when `"target_device": "CPU"` or `"target_device": "ANY"` set.

---

<a name="mixed_precision_quantization"></a>
#### Mixed-Precision Quantization

Quantization to lower precisions (e.g. 6, 4, 2 bits) is an efficient way to accelerate inference of neural networks.
Although NNCF supports quantization with an arbitrary number of bits to represent weights and activations values,
choosing ultra-low bitwidth could noticeably affect the model's accuracy. A good trade-off between accuracy and performance is achieved by assigning different precisions to different layers. NNCF provides two automatic precision assignment algorithms, namely **HAWQ** and **AutoQ**.

#### HAWQ
NNCF utilizes the [HAWQ-v2](https://arxiv.org/pdf/1911.03852.pdf) method to automatically choose optimal mixed-precision
configuration by taking into account the sensitivity of each layer, i.e. how much lower-bit quantization of each layer
decreases the accuracy of model. The most sensitive layers are kept at higher precision. The sensitivity of the i-th layer is
calculated by multiplying the average Hessian trace with the L2 norm of quantization perturbation:

![\overline{Tr}(H_{i}) * \left \| Q(W_{i}) - W_{i} \right \|^2_2](https://latex.codecogs.com/png.latex?%5Coverline%7BTr%7D%28H_%7Bi%7D%29%20*%20%5Cleft%20%5C%7C%20Q%28W_%7Bi%7D%29%20-%20W_%7Bi%7D%20%5Cright%20%5C%7C%5E2_2)

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
configuration file.

To avoid the exponential search procedure, we apply the following restriction: layers with a small average Hessian
trace value are quantized to lower bitwidth and vice versa.

The Hessian trace is estimated with the randomized [Hutchinson algorithm](https://www.researchgate.net/publication/220432178_Randomized_Algorithms_for_Estimating_the_Trace_of_an_Implicit_Symmetric_Positive_Semi-Definite_Matrix).
Given Rademacher distributed random vector v, the trace of symmetric matrix H is equal to the estimation of a quadratic form:

![Tr(H) = \mathbb{E}[v^T H v]](https://latex.codecogs.com/png.latex?Tr%28H%29%20%3D%20%5Cmathbb%7BE%7D%5Bv%5ET%20H%20v%5D)

The randomized algorithm solves the expectation by Monte Carlo using sampling of v from its distribution, evaluating
the quadratic term, and averaging:

![Tr(H) \approx \frac{1}{m}\sum_{i=1}^{m}[v_i^T H v_i]](https://latex.codecogs.com/png.latex?Tr%28H%29%20%5Capprox%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Bv_i%5ET%20H%20v_i%5D)

Evaluation of the quadratic term happens by computing ![Hv](https://latex.codecogs.com/png.latex?Hv) - the result
of multiplication of the Hessian matrix with a given random vector v, without the explicit formation of the Hessian operator.
For gradient of the loss with respect to the i-th block ![g_i](https://latex.codecogs.com/png.latex?g_i) and for
a random vector v, which is independent of ![W_i](https://latex.codecogs.com/png.latex?W_i), we have the equation:

![\frac{\partial(g_i^T v)}{\partial  W_i} = H_i v](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%28g_i%5ET%20v%29%7D%7B%5Cpartial%20W_i%7D%20%3D%20H_i%20v)

where ![H_i](https://latex.codecogs.com/png.latex?H_i) is the Hessian matrix of loss with respect to
![W_i](https://latex.codecogs.com/png.latex?W_i). Hence ![Hv](https://latex.codecogs.com/png.latex?Hv) can be
computed by 2 backpropagation passes: first  - with respect to the loss and second - with respect to the product of the
gradients and a random vector.

The aforementioned procedure sets bitwidth for weight quantizers only. Bitwidth for activation quantizers is assigned
on the next step in two ways: strict or liberal. All quantizers between modules with quantizable inputs have the same
bitwidth in the strict mode. Liberal mode allows different precisions within the group. For both cases, bitwidth is
assigned based on the rules of the hardware config. If multiple variants are possible the minimal compatible bitwidth
is chosen. By default, liberal mode is used as it does not reject a large number of possible bitwidth settings.
The `bitwidth_assignment_mode` parameter can override it to the strict one.

For automatic mixed-precision selection it's recommended to use the following template of configuration file:
```
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
                "compression_ratio": 1.5,
            }
        }
    }
```

Note, optimizer parameters are model specific, this template contains optimal ones for ResNet-like models.

Here's an [example](../../examples/torch/classification/configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_hawq.json) of
using the template in the full configuration file.

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

#### AutoQ
NNCF provides an alternate mode, namely AutoQ, for mixed-precision automation. It is an AutoML-based technique that automatically learns the layer-wise bitwidth with explored experiences. Based on [HAQ](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf), AutoQ utilizes an actor-critic algorithm, Deep Deterministic Policy Gradient (DDPG) for efficient search over the bitwidth space. DDPG is trained in an episodic fashion, converging to a deterministic mixed-precision policy after a number of episodes. An episode is constituted by stepping, the DDPG transitions from quantizer to quantizer sequentially to predict a precision of a layer. Each quantizer essentially denotes a state in RL framework and it is represented by attributes of the associated layers. For example, a quantizer for 2D Convolution is represented by its quantizer Id (integer), input and output channel size, feature map dimension, stride size, if it is depthwise, number of parameters etc. It is recommended to check out ```_get_layer_attr``` in [```quantization_env.py```](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/automl/environment/quantization_env.py#L333) for the featurization of different network layer types.

When the agent enters a state/quantizer, it receives the state features and forward passes them through its network. The output of the forward pass is a scalar continuous action output which is subsequently mapped to the bitwidth options of the particular quantizer. The episode terminates after the prediction of the last quantizer and a complete layer-wise mixed-precision policy is obtained. To ensure a policy fits in the user-specified compression ratio, the policy is post processed by reducing the precision sequentially from the last quantizer until the compression ratio is met.

To evaluate the goodness of a policy, NNCF backend quantizes the workload accordingly and performs evaluation with the user-registered function. The evaluated score, together with the state embedding, predicted action are appended to an experience vault to serve for DDPG learning. The learning is carried out by sampling the data point from the experience vault for supervised training of the DDPG network. This process typically happens at a fixed interval. In the current implementation, it is performed after each episode evaluation. For bootstrapping, exploration and diversity of experience, noise is added to action output. As the episodic iterations progress, the noise magnitude is gradually reduced to zero, a deterministic mixed-precision policy is converged at the end of the episodes. NNCF currently keeps track of the best policy and uses it for fine tuning.

```
    "target_device": "VPU",
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "precision": {
                "type": "autoq",
                "bits": [2, 4, 8],
                "iter_number": 300,
                "compression_ratio": 0.15,
                "eval_subset_ratio": 0.20,
                "dump_init_precision_data": true
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

```
    def autoq_eval_fn(model, eval_loader):
        _, top5 = validate(eval_loader, model, criterion, config)
        return top5

    nncf_config = register_default_init_args(
            nncf_config, init_loader, criterion, train_criterion_fn,
            autoq_eval_fn, val_loader, config.device)
```

#### Batch-norm statistics adaptation

After the compression-related changes in the model have been committed, the statistics of the batchnorm layers
(per-channel rolling means and variances of activation tensors) can be updated by passing several batches of data
through the model before the fine-tuning starts. This allows to correct the compression-induced bias in the model
and reduce the corresponding accuracy drop even before model training. This option is common for quantization, magnitude
sparsity and filter pruning algorithms. It can be enabled by setting a non-zero value of `num_bn_adaptation_samples` in the
`batchnorm_adaptation` section of the `initializer` configuration (see example below).


**Quantization configuration file parameters**:
```
{
    "algorithm": "quantization",
    "initializer": {
        "range": {
            "num_init_samples": 256, // Number of samples from the training dataset to consume as sample model inputs for purposes of setting initial minimum and maximum quantization ranges
            "type": "min_max" // Type of the initializer - determines which statistics gathered during initialization will be used to initialize the quantization ranges. "mean_min_max" is used by default
        },
        "precision": {
            "type": "hawq", // Type of precision initialization - either "manual" or "hawq". With "manual", precisions are defined explicitly via "bitwidth_per_scope". With "hawq", these are determined automatically using the HAWQ algorithm.
            "bits": [4, 8], // A list of bitwidth to choose from when performing precision initialization. Overrides bitwidth constraints specified in `weight` and `activation` sections",
            "num_data_points": 100, // Number of data points to iteratively estimate Hessian trace, 100 by default.
            "iter_number": 200, // Maximum number of iterations of Hutchinson algorithm to estimate Hessian trace, 200 by default
            "tolerance": 1e-4, //  Minimum relative tolerance for stopping the Hutchinson algorithm. It's calculated between mean average trace from previous iteration and current one. 1e-4 by default
            "compression_ratio": 1.5, // The desired ratio between bits complexity of fully INT8 model and mixed-precision lower-bit one.
            "bitwidth_per_scope": [ // Manual settings for the quantizer bitwidths. Scopes are used to identify the weight quantizers. The same number of bits is assigned to adjacent activation quantizers. By default bitwidth is taken from global quantization parameters from `weights` and `activations` sections above
                [
                    4,
                    "InsertionType.NNCF_MODULE_PRE_OP MobileNetV2/Sequential[features]/InvertedResidual[16]/Sequential[conv]/NNCFConv2d[2]"
                ], // A tuple of a bitwidth and a scope
                [
                    4,
                    "TargetType.OPERATOR_POST_HOOK MobileNetV2/Sequential[features]/ConvBNReLU[0]/ReLU6[2]/hardtanh_0",
                ]
            ]
        }
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 2048, // Number of samples from the training dataset to pass through the model at initialization in order to update batchnorm statistics of the original model. The actual number of samples will be a closest multiple of the batch size.
        }
    }
    "weights": { // Constraints to be applied to model weights quantization only.
        "mode": "symmetric", // Mode of quantization
        "bits": 8, // Bitwidth to quantize to. It is intended to manually specify bitwidth for all weights. Can be overridden by the `bits` parameter from the `precision` initializer section. An error happens if it doesn't match a bitwidth constraints for module weight specified in the hardware configuration.
        "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
        "per_channel": false, // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)

        // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
        "ignored_scopes": []

        // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
        // "target_scopes": []
    },
    "activations": { // Constraints to be applied to model activations quantization only.
        "mode": "symmetric", // Mode of quantization
        "bits": 4, // Bitwidth to quantize to. It is intended to manually specify bitwidth for all activations. Can be overridden by the `bits` parameter from the `precision` initializer section. An error happens if it doesn't match a bitwidth constraints for module inputs specified in the hardware configuration.
        "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
        "per_channel": false, // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)

        // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
        "ignored_scopes": []

        // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
        // "target_scopes": []

        // Specifies points in the model which will share the same quantizer module for activations. This is helpful in case one and the same quantizer scale is required for inputs to the same operation. Each sub-array will define a group of activation quantizer insertion points that have to share a single actual quantization module, each entry in this subarray should correspond to exactly one node in the NNCF graph and the groups should not overlap. The finalquantizer for each sub-array will be associated with the first element of this sub-array.
        "linked_quantizer_scopes": []
    },
    "quantize_inputs": true, // Whether the model inputs should be immediately quantized prior to any other model operations."
    "scope_overrides": { // This option is used to specify overriding quantization constraints for specific scope, e.g. in case you need to quantize a single operation differently than the rest of the model.
        "{re}.*InvertedResidual.*": {
            "mode": "symmetric", // Mode of quantization
            "bits": 4, // Bitwidth to quantize to.
            "signed": true, // Whether to use signed or unsigned input/output values for quantization. If specified as unsigned and the input values during initialization have differing signs, will reset to performing signed quantization instead.
            "per_channel": false // Whether to quantize inputs per channel (i.e. per 0-th dimension for weight quantization,and per 1-st dimension for activation quantization)
        }
    },

    // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
    "ignored_scopes": [],

    // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
    // "target_scopes": [],

    // Determines how should the additional quantization operations be exported into the ONNX format. Set this to false for export to OpenVINO-supported FakeQuantize ONNX, or to true for export to ONNX standard QuantizeLinear-DequantizeLinear node pairs (8-bit quantization only in the latter case). Default: false
    "export_to_onnx_standard_ops": false,
}
```

***Per layer ranges initializations parameters***:
Per layer ranges initiaization can be enabled by specifying  in `"initializer"` section `"range"` as list of dictionaries in the following format:

```
{
    "range": [
        {
            "type": "min_max", // Type of the initializer - determines which statistics gathered during initialization will be used to initialize the quantization ranges for all modules specified by `"target_scopes"` or `"ignored_scopes"`.

            "num_init_samples": 256, // Number of samples from the training dataset to consume as sample model inputs for purposes of setting initial minimum and maximum quantization ranges

            "target_scopes": [], // A list of model control flow graph node scopes to be considered for this operation - functions as a 'allowlist'. Optional.
            "ignored_scopes": [], // A list of model control flow graph node scopes to be ignored for this operation - functions as a 'denylist'. Optional.
            "target_quantizer_group": "weights" // Type of quantizer group to which this initialization of ranges will be applied. Optional. (By default this initialization of ranges will be applied to weights and activations quantizers)
        },
        ...
    ]
}

```
Initialization of ranges defined in this way must specify an unambiguous initialization rule for each module.
