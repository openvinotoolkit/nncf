:py:mod:`nncf.quantization.advanced_parameters`
===============================================

.. py:module:: nncf.quantization.advanced_parameters

.. autoapi-nested-parse::

   Structures and functions for passing advanced parameters to NNCF post-training quantization APIs.




Classes
~~~~~~~

.. autoapisummary::

   nncf.quantization.advanced_parameters.OverflowFix
   nncf.quantization.advanced_parameters.QuantizationParameters
   nncf.quantization.advanced_parameters.AdvancedBiasCorrectionParameters
   nncf.quantization.advanced_parameters.AdvancedQuantizationParameters
   nncf.quantization.advanced_parameters.AdvancedAccuracyRestorerParameters




.. py:class:: OverflowFix

   Bases: :py:obj:`enum.Enum`

   This option controls whether to apply the overflow issue fix for the 8-bit
   quantization.

   8-bit instructions of older Intel CPU generations (based on SSE, AVX-2, and AVX-512
   instruction sets) suffer from the so-called saturation (overflow) issue: in some
   configurations, the output does not fit into an intermediate buffer and has to be
   clamped. This can lead to an accuracy drop on the aforementioned architectures.
   The fix set to use only half a quantization range to avoid overflow for specific
   operations.

   If you are going to infer the quantized model on the architectures with AVX-2, and
   AVX-512 instruction sets, we recommend using FIRST_LAYER option as lower aggressive
   fix of the overflow issue. If you still face significant accuracy drop, try using
   ENABLE, but this may get worse the accuracy.

   :param ENABLE: All weights of all types of Convolutions and MatMul operations
       are be quantized using a half of the 8-bit quantization range.
   :param FIRST_LAYER: Weights of the first Convolutions of each model inputs
       are quantized using a half of the 8-bit quantization range.
   :param DISABLE: All weights are quantized using the full 8-bit quantization range.


.. py:class:: QuantizationParameters

   Contains quantization parameters for weights or activations.

   :param num_bits: The number of bits to use for quantization.
   :type num_bits: Optional[int]
   :param mode: The quantization mode to use, such as 'symmetric', 'asymmetric', etc.
   :type mode: nncf.common.quantization.structs.QuantizationMode
   :param signedness_to_force: Whether to force the weights or activations to be
       signed (True), unsigned (False)
   :type signedness_to_force: Optional[bool]
   :param per_channel: True if per-channel quantization is used, and False if
       per-tensor quantization is used.
   :type per_channel: Optional[bool]
   :param narrow_range: Whether to use a narrow quantization range.

       If False, then the input will be quantized into quantization range

       * [0; 2^num_bits - 1] for unsigned quantization and
       * [-2^(num_bits - 1); 2^(num_bits - 1) - 1] for signed quantization

       If True, then the ranges would be:

       * [0; 2^num_bits - 2] for unsigned quantization and
       * [-2^(num_bits - 1) + 1; 2^(num_bits - 1) - 1] for signed quantization
   :type narrow_range: Optional[bool]


.. py:class:: AdvancedBiasCorrectionParameters

   Contains advanced parameters for fine-tuning bias correction algorithm.

   :param apply_for_all_nodes: Whether to apply the correction to all nodes in the
       model, or only to nodes that have a bias.
   :type apply_for_all_nodes: bool
   :param threshold: The threshold value determines the maximum bias correction value.
       The bias correction are skipped If the value is higher than threshold.
   :type threshold: Optional[float]


.. py:class:: AdvancedQuantizationParameters

   Contains advanced parameters for fine-tuning qunatization algorithm.

   :param overflow_fix: This option controls whether to apply the overflow issue fix
       for the 8-bit quantization, defaults to OverflowFix.FIRST_LAYER.
   :type overflow_fix: nncf.quantization.advanced_parameters.OverflowFix
   :param quantize_outputs: Whether to insert additional quantizers right before each
       of the model outputs.
   :type quantize_outputs: bool
   :param inplace_statistics: Defines whether to calculate quantizers statistics by
       backend graph operations or by default Python implementation, defaults to True.
   :type inplace_statistics: bool
   :param disable_bias_correction: Whether to disable the bias correction.
   :type disable_bias_correction: bool
   :param activations_quantization_params: Quantization parameters for activations.
   :type activations_quantization_params: nncf.quantization.advanced_parameters.QuantizationParameters
   :param weights_quantization_params: Quantization parameters for weights.
   :type weights_quantization_params: nncf.quantization.advanced_parameters.QuantizationParameters
   :param activations_range_estimator_params: Range estimator parameters for activations.
   :type activations_range_estimator_params: nncf.quantization.range_estimator.RangeEstimatorParameters
   :param weights_range_estimator_params: Range estimator parameters for weights.
   :type weights_range_estimator_params: nncf.quantization.range_estimator.RangeEstimatorParameters
   :param bias_correction_params: Advanced bias correction parameters.
   :type bias_correction_params: nncf.quantization.advanced_parameters.AdvancedBiasCorrectionParameters
   :param backend_params: Backend-specific parameters.
   :type backend_params: Dict[str, Any]


.. py:class:: AdvancedAccuracyRestorerParameters

   Contains advanced parameters for fine-tuning the accuracy restorer algorithm.

   :param max_num_iterations: The maximum number of iterations of the algorithm.
       In other words, the maximum number of layers that may be reverted back to
       floating-point precision. By default, it is limited by the overall number of
       quantized layers.
   :type max_num_iterations: int
   :param tune_hyperparams: Whether to tune of quantization parameters as a
       preliminary step before reverting layers back to the floating-point precision.
       It can bring an additional boost in performance and accuracy, at the cost of
       increased overall quantization time. The default value is `False`.
   :type tune_hyperparams: int
   :param convert_to_mixed_preset: Whether to convert the model to mixed mode if
       the accuracy criteria of the symmetrically quantized model are not satisfied.
       The default value is `False`.
   :type convert_to_mixed_preset: bool
   :param ranking_subset_size: Size of a subset that is used to rank layers by their
       contribution to the accuracy drop.
   :type ranking_subset_size: Optional[int]


