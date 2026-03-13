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

## Gradient Computation for Quantization-Aware Training

The forward quantization formula contains two non-differentiable operations: clamping and rounding. To enable gradient-based optimization of the quantization parameters during QAT, NNCF defines custom surrogate gradients using a **Straight-Through Estimator (STE)** for rounding and piecewise-defined surrogate gradients for the clamp boundaries.

This approach is a form of **learned-range fake quantization** — it is related to [Learned Step Size Quantization (LSQ)](https://arxiv.org/abs/1902.08153), but uses a different parameterization (`input_low`, `input_range`) instead of (step size, zero point), and omits LSQ's gradient scaling factor.

In this section, $x$ denotes an element of the input tensor and $FQ(x)$ denotes the corresponding fake-quantized output. The quantization parameters $input\\_low$ and $input\\_range$ are the values that enter the forward kernel (i.e. after the absolute-value and range-tuning steps described above), so $input\\_range > 0$. We write $s = (levels - 1) / input\\_range$ for the scale factor.

### Input Partitioning

The input tensor is partitioned element-wise into three regions based on the quantization range:

- **Below range**: $x < input\\_low$
- **In range**: $input\\_low \le x \le input\\_low + input\\_range$
- **Above range**: $x > input\\_low + input\\_range$

### Gradient w.r.t. $x$ (STE)

The upstream gradient is passed through unchanged when $x$ is within the quantization range, and zeroed out otherwise:

$$
\frac{\partial \mathcal{L}}{\partial x} = \begin{cases}
\dfrac{\partial \mathcal{L}}{\partial FQ} & \text{in range} \\[6pt]
0 & \text{below or above range}
\end{cases}
$$

### Gradient w.r.t. $input\\_range$

The gradient with respect to $input\\_range$ depends on which region the input falls in. Per-element gradients are summed to match the shape of $input\\_range$ (scalar for per-tensor quantization, or per-channel).

$$
\frac{\partial \mathcal{L}}{\partial input\_range} = \begin{cases}
\dfrac{\partial \mathcal{L}}{\partial FQ} \cdot \dfrac{FQ(x) - x}{input\_range} & \text{in range} \\[10pt]
\dfrac{\partial \mathcal{L}}{\partial FQ} & \text{above range} \\[10pt]
\dfrac{\partial \mathcal{L}}{\partial FQ} \cdot \dfrac{level\_low}{level\_high} & \text{below range}
\end{cases}
$$

#### Derivation of the in-range term

For in-range $x$, the forward pass is:

$$
FQ(x) = \frac{\left\lfloor (x - input\_low) \cdot s \;-\; ZP \right\rceil}{s}
$$

where $ZP = \lfloor -input\_low \cdot s \rceil$. The STE treats each rounding as identity plus a constant residual: $\lfloor u \rceil = u + \epsilon$ where $\epsilon = \lfloor u \rceil - u$ is held constant during differentiation. Applying this to $ZP$ and then to the outer rounding, the two $input\_low \cdot s$ contributions cancel and we obtain:

$$
FQ(x) \;\approx\; \frac{x \cdot s \;+\; \epsilon}{s}
\;=\; x \;+\; \frac{\epsilon}{s}
$$

where $\epsilon$ is the combined residual from both rounding operations. The $x$ term is independent of $input\_range$; the $\epsilon / s$ term depends on it through $1/s = input\_range / (levels - 1)$, with $\epsilon$ treated as constant:

$$
\frac{\partial FQ}{\partial input\_range}
= \frac{\epsilon}{levels - 1}
$$

To re-express $\epsilon$ in terms of knowable quantities: from the STE expansion above, $\epsilon = (FQ(x) - x) \cdot s$. Substituting:

$$
\frac{\partial FQ}{\partial input\_range}
= \frac{(FQ(x) - x) \cdot s}{levels - 1}
= \frac{FQ(x) - x}{input\_range}
$$

This gradient nudges $input\\_range$ to reduce quantization error: if $FQ(x) > x$, the gradient encourages shrinking $input\\_range$ (finer step size), and vice versa.

#### Above-range term

For $x > input\\_low + input\\_range$, the output is clamped: $FQ(x) = input\\_low + input\\_range$, so $\partial FQ / \partial input\\_range = 1$.

#### Below-range term

For $x < input\\_low$, the output is clamped to $input\\_low$, which does not depend on $input\\_range$ (in asymmetric mode). The code uses the surrogate gradient $\alpha = level\\_low / level\\_high$ rather than the analytic derivative (which would be $0$). For asymmetric quantization ($level\\_low = 0$), this gives $\alpha = 0$. For symmetric quantization with signed range ($level\\_low < 0$), the non-zero $\alpha$ matches the analytic derivative of $input\\_low$ with respect to the $scale$ parameter, since $input\\_low = scale \cdot level\\_low / level\\_high$ in symmetric mode.

### Gradient w.r.t. $input\\_low$

$$
\frac{\partial \mathcal{L}}{\partial input\_low} = \begin{cases}
0 & \text{in range} \\[6pt]
\dfrac{\partial \mathcal{L}}{\partial FQ} & \text{below or above range}
\end{cases}
$$

Per-element gradients are summed to match the shape of $input\\_low$.

**In-range term.** Under the STE, shifting $input\\_low$ moves the quantization grid, but the zero-point $ZP = \lfloor -input\\_low \cdot s \rceil \approx -input\\_low \cdot s$ shifts to compensate. In the STE expansion above, these two contributions cancel (the $-input\\_low \cdot s$ and $+input\\_low \cdot s$ terms), making $FQ(x) \approx x + \epsilon/s$ with no dependence on $input\\_low$. This is unlike $input\\_range$, which affects the step size $1/s$ and therefore scales the rounding residual $\epsilon$.

**Below- and above-range terms.** Outside the range, the clamped output is either $input\\_low$ (below) or $input\\_low + input\\_range$ (above), both of which have $\partial / \partial input\\_low = 1$.

**Note:** In symmetric quantization mode, $input\\_low$ is derived from $input\\_range$ (i.e. $scale$) and is not an independent learnable parameter, so its gradient is not used directly.

---

**NOTE**

There is a known issue with AVX2 and AVX512 CPU devices. The issue appears with 8-bit matrix calculations with tensors which elements are close to the maximum or saturated.
AVX2 and AVX512 utilize a 16-bit register to store the result of operations on tensors. In case when tensors are saturated the buffer overflow happens.
This leads to accuracy degradation. For more details of the overflow issue please refer [here](https://www.intel.com/content/www/us/en/developer/articles/technical/lower-numerical-precision-deep-learning-inference-and-training.html).

To fix this issue inside NNCF, by default, all weight tensors are quantized in 8 bits but only 7 bits are effectively used.
This regime is used when `target_device=TargetDevice.CPU` or `target_device=TargetDevice.ANY` set. This fix, potentially, requires longer fine-tuning.

To control the application of overflow fix, `nncf.AdvancedQuantizationParameters(overflow_fix=OverflowFix.ENABLE)` config option is introduced.
