:orphan:

:py:mod:`nncf.quantization.advanced_parameters`
===============================================

.. py:module:: nncf.quantization.advanced_parameters



Classes
~~~~~~~

.. autoapisummary::

   nncf.quantization.advanced_parameters.AdvancedQuantizationParameters




.. py:class:: AdvancedQuantizationParameters

   Contains advanced parameters for fine-tuning qunatization algorithm.

   :param overflow_fix: This option controls whether to apply the overflow issue fix
       for the 8-bit quantization, defaults to OverflowFix.FIRST_LAYER.
   :param quantize_outputs: Whether to insert additional quantizers right before each
       of the model outputs.
   :param inplace_statistics: Defines wheather to calculate quantizers statistics by
       backend graph operations or by default Python implementation, defaults to True.
   :param disable_bias_correction: Whether to disable the bias correction.
   :param activations_quantization_params: Quantization parameters for activations.
   :param weights_quantization_params: Quantization parameters for weights.
   :param activations_range_estimator_params: Range estimator parameters for
       activations.
   :param weights_range_estimator_params: Range estimator parameters for weights.
   :param bias_correction_params: Advanced bias correction paramters.
   :param backend_params: Backend-specific parameters.


