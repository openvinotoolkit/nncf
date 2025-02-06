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
"""
Structures and functions for passing advanced parameters to NNCF post-training quantization APIs.
"""

import sys
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import nncf
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationRule
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.utils.api_marker import api
from nncf.parameters import StrEnum
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import StatisticsType


@api(canonical_alias="nncf.OverflowFix")
class OverflowFix(StrEnum):
    """
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
    """

    ENABLE = "enable"
    FIRST_LAYER = "first_layer_only"
    DISABLE = "disable"


@api()
class FP8Type(StrEnum):
    """
    Defines FP8 special types (https://arxiv.org/pdf/2209.05433.pdf).

    :param E4M3: Mode with 4-bit exponent and 3-bit mantissa.
    :param E5M2: Mode with 5-bit exponent and 2-bit mantissa.

    """

    E4M3 = "f8e4m3"
    E5M2 = "f8e5m2"


@api()
@dataclass
class QuantizationParameters:
    """
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
    """

    num_bits: Optional[int] = None
    mode: Optional[QuantizationMode] = None
    signedness_to_force: Optional[bool] = None
    per_channel: Optional[bool] = None
    narrow_range: Optional[bool] = None


@api()
@dataclass
class FP8QuantizationParameters:
    """
    Contains convert parameters for weights or activations.

    :param destination_type: Currently contains E4M3 or E5M2 for FP8 precision.
    :type destination_type: FP8Type
    """

    destination_type: Optional[FP8Type] = None


@api()
@dataclass
class AdvancedBiasCorrectionParameters:
    """
    Contains advanced parameters for fine-tuning bias correction algorithm.

    :param apply_for_all_nodes: Whether to apply the correction to all nodes in the
        model, or only to nodes that have a bias.
    :type apply_for_all_nodes: bool
    :param threshold: The threshold value determines the maximum bias correction value.
        The bias correction are skipped If the value is higher than threshold.
    :type threshold: Optional[float]
    """

    apply_for_all_nodes: bool = False
    threshold: Optional[float] = None


@api()
@dataclass
class AdvancedSmoothQuantParameters:
    """
    Contains advanced alpha parameters for SmoothQuant algorithm.
    It regulates the calculation of the smooth scale for different node types.
    A negative value switches off the algorithm for current node type. In case of inaccurate results,
    this parameter may be adjusted in the range from 0 to 1 or set -1 to disable SmoothQuant algorithm.

    :param convolution: Whether to apply smoothing for Convolution layers.
    :type convolution: float
    :param matmul: Whether to apply smoothing for MatMul layers.
    :type matmul: float
    """

    convolution: float = -1
    matmul: float = 0.95


class RestoreMode(Enum):
    """
    Specifies how to revert operations to their original precision.

    :param ACTIVATIONS_AND_WEIGHTS: Operations will be reverted to floating-point precision.
    :param ONLY_ACTIVATIONS: Operations with weights will be reverted to representation with int8 weights,
        while all other operations will revert to floating-point precision.
    """

    ACTIVATIONS_AND_WEIGHTS = "activations_and_weights"
    ONLY_ACTIVATIONS = "only_activations"


@api()
@dataclass
class AdvancedQuantizationParameters:
    """
    Contains advanced parameters for fine-tuning quantization algorithm.

    :param overflow_fix: This option controls whether to apply the overflow issue fix
        for the 8-bit quantization.
    :type overflow_fix: Optional[nncf.quantization.advanced_parameters.OverflowFix]
    :param quantize_outputs: Whether to insert additional quantizers right before each
        of the model outputs.
    :type quantize_outputs: bool
    :param inplace_statistics: Defines whether to calculate quantizers statistics by
        backend graph operations or by default Python implementation, defaults to True.
    :type inplace_statistics: bool
    :param disable_channel_alignment: Whether to disable the channel alignment.
    :type disable_channel_alignment: bool
    :param disable_bias_correction: Whether to disable the bias correction.
    :type disable_bias_correction: bool
    :param batchwise_statistics: Determines whether quantizer statistics should be calculated
        for each item of the batch or for the entire batch, default is None.
        "None" means that if torch.DataLoader or tensorflow.Dataset was passed as a data source for
        the calibration dataset, then in case batch_size of the data source > 1 batchwise_statistics sets to True,
        otherwise sets to False.
    :type batchwise_statistics: Optional[bool]
    :param quantizer_propagation_rule: An instance of the `QuantizerPropagationRule` enum that
        specifies how quantizers should be propagated and merged across branching nodes in the
        model's computational graph. The strategies are as follows:
        - DO_NOT_MERGE_BRANCHES: No merging of quantization parameters across branches.
        - MERGE_IF_ALL_BRANCHES_SAME : Merge only if all branch quantization configurations are identical.
        - MERGE_WITH_POTENTIAL_REQUANTIZATION: Merge common configurations and allow for requantization
        on branches with additional options.
        - MERGE_ALL_IN_ONE: Attempt to merge into a single global quantization configuration
        if possible given hardware constraints.
        MERGE_ALL_IN_ONE is a default value.
    :type quantizer_propagation_rule: QuantizerPropagationRule
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
    :param smooth_quant_alphas: SmoothQuant-related parameters mapping.
        It regulates the calculation of the smooth scale. The default value stored in AdvancedSmoothQuantParameters.
        A negative value for each field switches off type smoothing. In case of inaccurate results,
        fields may be adjusted in the range from 0 to 1 or set -1 to disable smoothing for type.
    :type smooth_quant_alphas: nncf.quantization.advanced_parameters.AdvancedSmoothQuantParameters
    :param smooth_quant_alpha: Deprecated SmoothQuant-related parameter.
    :type smooth_quant_alpha: float
    :param backend_params: Backend-specific parameters.
    :type backend_params: Dict[str, Any]
    """

    # General parameters
    overflow_fix: Optional[OverflowFix] = None
    quantize_outputs: bool = False
    inplace_statistics: bool = True
    disable_channel_alignment: bool = True
    disable_bias_correction: bool = False
    batchwise_statistics: Optional[bool] = None

    # Advanced Quantization parameters
    activations_quantization_params: Optional[Union[QuantizationParameters, FP8QuantizationParameters]] = None
    weights_quantization_params: Optional[Union[QuantizationParameters, FP8QuantizationParameters]] = None
    quantizer_propagation_rule: QuantizerPropagationRule = QuantizerPropagationRule.MERGE_ALL_IN_ONE

    # Range estimator parameters
    activations_range_estimator_params: RangeEstimatorParameters = field(default_factory=RangeEstimatorParameters)
    weights_range_estimator_params: RangeEstimatorParameters = field(default_factory=RangeEstimatorParameters)

    # Advanced BiasCorrection algorithm parameters
    bias_correction_params: AdvancedBiasCorrectionParameters = field(default_factory=AdvancedBiasCorrectionParameters)

    # Advanced SmoothQuant algorithm parameters
    smooth_quant_alphas: AdvancedSmoothQuantParameters = field(default_factory=AdvancedSmoothQuantParameters)
    # Deprecated parameter
    smooth_quant_alpha: Optional[float] = None

    # Backend specific parameters
    backend_params: Dict[str, Any] = field(default_factory=dict)


@api()
@dataclass
class AdvancedAWQParameters:
    """
    Contains advanced parameters for AWQ algorithm.

    :param subset_size: The number of samples for AWQ.
    :type subset_size: int
    :param percent_to_apply: The percent of outliers for correction.
    :type percent_to_apply: float
    :param alpha_min: Minimum value of smoothness parameter for grid search.
    :type alpha_min: float
    :param alpha_max: Maximal value of smoothness parameter for grid search.
    :type alpha_max: float
    :param steps: The number of the steps in grid search.
    :type steps: int
    """

    subset_size: int = 32
    percent_to_apply: float = 0.002
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    steps: int = 100


@api()
@dataclass
class AdvancedScaleEstimationParameters:
    """
    Contains advanced parameters for scale estimation algorithm.

    :param subset_size: The number of samples for scale estimation.
    :type subset_size: int
    :param initial_steps: The number of the steps for absmax scale rectification.
    :type initial_steps: int
    :param scale_steps: The number of the steps for grid search scale rectification
        from 1.0 to 1.0 - 0.05 * scale_step.
    :type scale_steps: int
    :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
    :type weight_penalty: float
    """

    subset_size: int = 64
    initial_steps: int = 5
    scale_steps: int = 5
    weight_penalty: float = -1.0


@api()
@dataclass
class AdvancedGPTQParameters:
    """
    Contains advanced parameters for GPTQ algorithm.

    :param damp_percent: The percent of the average Hessian diagonal to use for dampening,
        recommended value is 0.1.
    :type damp_percent: float
    :param block_size: The size of the blocks used during quantization. Defaults to 128.
    :type block_size: int
    :param subset_size: Number of data samples to calculate Hessian. Defaults to 128.
    :type subset_size: int
    """

    damp_percent: float = 0.1
    block_size: int = 128
    subset_size: int = 128


@api()
@dataclass
class AdvancedLoraCorrectionParameters:
    """
    Contains advanced parameters for lora correction algorithm.

    :param adapter_rank: rank of lora adapters. Defaults to 16.
    :type adapter_rank: int
    :param num_iterations: number of correction iterations. Defaults to 3.
    :type num_iterations: int
    :param apply_regularization: Whether to add a regularization during the correction process. Defaults to True.
        Helpful for big rank values to avoid overfitting.
    :type apply_regularization: bool
    :param subset_size: Number of data samples for lora correction algorithm. Defaults to 128.
    :type subset_size: int
    :param use_int8_adapters: Whether to 8-bit quantize lora adapters, otherwise they kept in the original weights
        precision. Defaults to True.
    :type use_int8_adapters: bool
    """

    adapter_rank: int = 8
    num_iterations: int = 3
    apply_regularization: bool = True
    subset_size: int = 128
    use_int8_adapters: bool = True


@api()
@dataclass
class AdvancedCompressionParameters:
    """
    Contains advanced parameters for compression algorithms.

    :param statistics_path: Directory path to dump statistics.
    :type statistics_path: str
    :param awq_params: Advanced parameters for AWQ algorithm.
    :type awq_params: AdvancedAWQParameters
    :param scale_estimation_params: Advanced parameters for scale estimation algorithm.
    :type scale_estimation_params: AdvancedScaleEstimationParameters
    """

    statistics_path: Optional[str] = None
    # Advanced AWQ algorithm parameters
    awq_params: AdvancedAWQParameters = field(default_factory=AdvancedAWQParameters)

    # Advanced scale estimation algorithm parameters
    scale_estimation_params: AdvancedScaleEstimationParameters = field(
        default_factory=AdvancedScaleEstimationParameters
    )

    # Advanced GPTQ algorithm parameters
    gptq_params: AdvancedGPTQParameters = field(default_factory=AdvancedGPTQParameters)

    # Advanced Lora Correction algorithm parameters
    lora_correction_params: AdvancedLoraCorrectionParameters = field(default_factory=AdvancedLoraCorrectionParameters)


@api()
@dataclass
class AdvancedAccuracyRestorerParameters:
    """
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
    :param ranking_subset_size: Size of a subset that is used to rank layers by their
        contribution to the accuracy drop.
    :type ranking_subset_size: Optional[int]
    :param num_ranking_workers: The number of parallel workers that are used to rank
        quantization operations.
    :type num_ranking_workers: Optional[int]
    :param intermediate_model_dir: Path to the folder where the model, which was fully
        quantized with initial parameters, should be saved.
    :type intermediate_model_dir: Optional[str]
    :param restore_mode: Specifies how to revert operations to their original precision.
    :type restore_mode: RestoreMode
    """

    max_num_iterations: int = sys.maxsize
    tune_hyperparams: bool = False
    ranking_subset_size: Optional[int] = None
    num_ranking_workers: Optional[int] = None
    intermediate_model_dir: Optional[str] = None
    restore_mode: RestoreMode = RestoreMode.ACTIVATIONS_AND_WEIGHTS


def changes_asdict(params: Any) -> Dict[str, Any]:
    """
    Returns non None fields as dict

    :param params: A dataclass instance
    :return: A dict with non None fields
    """
    changes = {}
    for f in fields(params):
        value = getattr(params, f.name)
        if value is not None:
            changes[f.name] = value
    return changes


def convert_to_dict_recursively(params: Any) -> Dict[str, Any]:
    """
    Converts dataclass to dict recursively

    :param params: A dataclass instance
    :return: A dataclass as dict
    """
    if params is None:
        return {}

    result = {}
    for f in fields(params):
        value = getattr(params, f.name)
        if is_dataclass(value):
            result[f.name] = convert_to_dict_recursively(value)
        elif isinstance(value, Enum):
            result[f.name] = value.value
        else:
            result[f.name] = value

    return result


def convert_quantization_parameters_to_dict(params: Optional[QuantizationParameters]) -> Dict[str, Any]:
    """
    Converts quantization parameters to the dict in the legacy format

    :param params: Quantization parameters
    :return: Quantization parameters as dict in the legacy format
    """
    result: Dict[str, Any] = {}
    if params is not None:
        if params.num_bits is not None:
            result["bits"] = params.num_bits
        if params.mode is not None:
            result["mode"] = params.mode
        if params.signedness_to_force is not None:
            result["signed"] = params.signedness_to_force
        if params.per_channel is not None:
            result["per_channel"] = params.per_channel
        if params.narrow_range is not None:
            raise nncf.ParameterNotSupportedError("narrow_range parameter is not supported in the legacy format")
    return result


def convert_range_estimator_parameters_to_dict(params: RangeEstimatorParameters) -> Dict[str, Any]:
    """
    Converts range estimator parameters to the dict in the legacy format

    :param params: Range estimator parameters
    :return: range estimator parameters as dict in the legacy format
    """
    if params.min.clipping_value is not None or params.max.clipping_value is not None:
        raise nncf.ParameterNotSupportedError("clipping_value parameter is not supported in the legacy format")

    result: Dict[str, Any] = {}
    if (
        params.min.statistics_type == StatisticsType.MIN
        and params.min.aggregator_type == AggregatorType.MIN
        and params.max.statistics_type == StatisticsType.MAX
        and params.max.aggregator_type == AggregatorType.MAX
    ):
        result["type"] = "mixed_min_max"
    elif (
        params.min.statistics_type == StatisticsType.MIN
        and params.min.aggregator_type == AggregatorType.MEAN
        and params.max.statistics_type == StatisticsType.MAX
        and params.max.aggregator_type == AggregatorType.MEAN
    ):
        result["type"] = "mean_min_max"
    elif (
        params.min.statistics_type == StatisticsType.QUANTILE
        and params.min.aggregator_type == AggregatorType.MEAN
        and params.max.statistics_type == StatisticsType.QUANTILE
        and params.max.aggregator_type == AggregatorType.MEAN
    ):
        result["type"] = "mean_percentile"
        result["params"] = {
            "min_percentile": 1 - params.min.quantile_outlier_prob,
            "max_percentile": 1 - params.max.quantile_outlier_prob,
        }
    elif (
        params.min.statistics_type is None
        and params.min.aggregator_type is None
        and params.max.statistics_type is None
        and params.max.aggregator_type is None
    ):
        return {}
    else:
        raise nncf.ParameterNotSupportedError(
            f"The following range estimator parameters are not supported: {str(params)}"
        )

    return result


def apply_advanced_parameters_to_config(
    config: Dict[str, Any], params: AdvancedQuantizationParameters
) -> Dict[str, Any]:
    """
    Apply advanced parameters to the config in the legacy format

    :param config: NNCF config in legacy format
    :param params: Advanced quantization parameters
    :return: advanced quantization parameters as dict in the legacy format
    """
    config["overflow_fix"] = params.overflow_fix if params.overflow_fix is None else params.overflow_fix.value
    config["quantize_outputs"] = params.quantize_outputs

    if params.disable_bias_correction:
        initializer = config.get("initializer", {})
        initializer["batchnorm_adaptation"] = {"num_bn_adaptation_samples": 0}
        config["initializer"] = initializer

    if isinstance(params.activations_quantization_params, QuantizationParameters):
        activations_config = convert_quantization_parameters_to_dict(params.activations_quantization_params)
        if activations_config:
            config["activations"] = activations_config

    if isinstance(params.weights_quantization_params, QuantizationParameters):
        weights_config = convert_quantization_parameters_to_dict(params.weights_quantization_params)
        if weights_config:
            config["weights"] = weights_config

    activations_init_range_config = convert_range_estimator_parameters_to_dict(
        params.activations_range_estimator_params
    )
    weights_init_range_config = convert_range_estimator_parameters_to_dict(params.weights_range_estimator_params)

    if activations_init_range_config or weights_init_range_config:
        initializer = config.get("initializer", {})
        init_range = initializer.get("range", {})
        global_num_init_samples = init_range.get("num_init_samples", None)
        global_range_type = init_range.get("type", None)

        activations_init_range_config["target_quantizer_group"] = "activations"
        activations_init_range_config["target_scopes"] = "{re}.*"
        if global_num_init_samples is not None:
            activations_init_range_config["num_init_samples"] = global_num_init_samples
        if "type" not in activations_init_range_config and global_range_type is not None:
            activations_init_range_config["type"] = global_range_type

        weights_init_range_config["target_quantizer_group"] = "weights"
        weights_init_range_config["target_scopes"] = "{re}.*"
        if global_num_init_samples is not None:
            weights_init_range_config["num_init_samples"] = global_num_init_samples
        if "type" not in weights_init_range_config and global_range_type is not None:
            weights_init_range_config["type"] = global_range_type

        initializer["range"] = [activations_init_range_config, weights_init_range_config]
        config["initializer"] = initializer

    if params.bias_correction_params.apply_for_all_nodes:
        raise nncf.ParameterNotSupportedError(
            "apply_for_all_nodes parameter of the BiasCorrection algorithm is not supported in the legacy format"
        )

    if params.bias_correction_params.threshold is not None:
        raise nncf.ParameterNotSupportedError(
            "threshold parameter of the BiasCorrection algorithm is not supported in the legacy format"
        )

    return config
