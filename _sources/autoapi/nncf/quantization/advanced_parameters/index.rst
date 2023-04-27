:py:mod:`nncf.quantization.advanced_parameters`
===============================================

.. py:module:: nncf.quantization.advanced_parameters

.. autoapi-nested-parse::

   Copyright (c) 2023 Intel Corporation
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.




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


