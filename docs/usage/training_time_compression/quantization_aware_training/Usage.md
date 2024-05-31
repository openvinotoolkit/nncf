# Use NNCF for Quantization Aware Training

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

## Quantization-aware Training Implementation

- In [PyTorch](Quantization.md)
- In [TensorFlow](../other_algorithms/LegacyQuantization.md)
