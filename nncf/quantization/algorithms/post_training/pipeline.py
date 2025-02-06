# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, TypeVar

from nncf.common.deprecation import warning_deprecated
from nncf.common.quantization.structs import QuantizationPreset
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.bias_correction.algorithm import BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.channel_alignment.algorithm import ChannelAlignment
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FAST_BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.pipeline import Pipeline
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")


def create_ptq_pipeline(
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> Pipeline:
    """
    Creates a post-training quantization pipeline.

    The post-training quantization pipeline includes the following steps:
        1) SmoothQuant
        2) ChannelAlignment
        3) MinMaxQuantization
        4) FastBiasCorrection or BiasCorrection

    :param mode: Special quantization mode that specify different ways of the optimization.
    :param preset: A preset controls the quantization mode (symmetric and asymmetric).
        It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
        Default value is None. In this case, `mixed` preset is used for `transformer`
        model type otherwise `performace`.
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_bias_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param advanced_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm
    :return: A post-training quantization pipeline.
    """

    if advanced_parameters is None:
        advanced_parameters = AdvancedQuantizationParameters()

    # Build the post-training quantization pipeline.
    pipeline_steps = []

    # Add the `SmoothQuant` algorithm as the first step of the pipeline.
    # It is added only for `ModelType.TRANSFORMER`.
    sq_params = advanced_parameters.smooth_quant_alphas
    sq_alpha = advanced_parameters.smooth_quant_alpha
    if sq_alpha is not None:
        warning_deprecated(
            "`AdvancedQuantizationParameters(smooth_quant_alpha=..)` is deprecated."
            "Please, use `AdvancedQuantizationParameters(smooth_quant_alphas)` option "
            "with AdvancedSmoothQuantParameters(convolution=.., matmul=..) as value instead."
        )
        if sq_alpha < 0:
            sq_params.convolution = -1
            sq_params.matmul = -1
        else:
            sq_params.matmul = sq_alpha

    if model_type == ModelType.TRANSFORMER and (sq_params.convolution >= 0 or sq_params.matmul >= 0):
        alpha_map = {"convolution": sq_params.convolution, "matmul": sq_params.matmul}
        pipeline_steps.append([SmoothQuant(subset_size, advanced_parameters.inplace_statistics, alpha_map=alpha_map)])

    # Add the `ChannelAlignment` algorithm as the second step of the pipeline.
    if not advanced_parameters.disable_channel_alignment:
        pipeline_steps.append([ChannelAlignment(subset_size, advanced_parameters.inplace_statistics)])

    # Add the `MinMaxQuantization` algorithm as the third step of the pipeline.
    pipeline_steps.append(
        [
            MinMaxQuantization(
                mode=mode,
                preset=preset,
                target_device=target_device,
                subset_size=subset_size,
                model_type=model_type,
                ignored_scope=ignored_scope,
                overflow_fix=advanced_parameters.overflow_fix,
                quantize_outputs=advanced_parameters.quantize_outputs,
                inplace_statistics=advanced_parameters.inplace_statistics,
                batchwise_statistics=advanced_parameters.batchwise_statistics,
                activations_quantization_params=advanced_parameters.activations_quantization_params,
                weights_quantization_params=advanced_parameters.weights_quantization_params,
                activations_range_estimator_params=advanced_parameters.activations_range_estimator_params,
                weights_range_estimator_params=advanced_parameters.weights_range_estimator_params,
                quantizer_propagation_rule=advanced_parameters.quantizer_propagation_rule,
                backend_params=advanced_parameters.backend_params,
            )
        ]
    )

    if not advanced_parameters.disable_bias_correction:
        # Add the `FastBiasCorrection` or `BiasCorrection` as additional algorithm
        # inside the third step of the pipeline. It is added after `MinMaxQuantization`
        # algorithm.
        bias_correction_params = advanced_parameters.bias_correction_params
        if fast_bias_correction:
            threshold = FAST_BIAS_CORRECTION_THRESHOLD
            bias_correction_subset_size = subset_size
            bias_correction_cls = FastBiasCorrection
        else:
            threshold = BIAS_CORRECTION_THRESHOLD
            bias_correction_subset_size = max(int(subset_size * 0.2), 1)
            bias_correction_cls = BiasCorrection

        if bias_correction_params.threshold is not None:
            threshold = bias_correction_params.threshold

        pipeline_steps[-1].append(
            bias_correction_cls(
                bias_correction_subset_size,
                threshold,
                bias_correction_params.apply_for_all_nodes,
                advanced_parameters.inplace_statistics,
                advanced_parameters.backend_params,
            )
        )

    return Pipeline(pipeline_steps)
