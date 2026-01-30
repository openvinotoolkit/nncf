# Uniform Quantization with Fine-Tuning

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

---

**NOTE**

There is a known issue with AVX2 and AVX512 CPU devices. The issue appears with 8-bit matrix calculations with tensors which elements are close to the maximum or saturated.
AVX2 and AVX512 utilize a 16-bit register to store the result of operations on tensors. In case when tensors are saturated the buffer overflow happens.
This leads to accuracy degradation. For more details of the overflow issue please refer [here](https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html).

To fix this issue inside NNCF, by default, all weight tensors are quantized in 8 bits but only 7 bits are effectively used.
This regime is used when `target_device=TargetDevice.CPU` or `target_device=TargetDevice.ANY` set. This fix, potentially, requires longer fine-tuning.

To control the application of overflow fix, `nncf.AdvancedQuantizationParameters(overflow_fix=OverflowFix.ENABLE)` config option is introduced.
