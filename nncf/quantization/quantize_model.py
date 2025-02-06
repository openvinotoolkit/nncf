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

from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import nncf
from nncf.api.compression import TModel
from nncf.common.deprecation import warning_deprecated
from nncf.common.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging.logger import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.parameters import BackupMode
from nncf.parameters import CompressWeightsMode
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import HyperparameterTuner
from nncf.quantization.algorithms.hyperparameter_tuner.param_grid import get_quantization_param_grids
from nncf.quantization.algorithms.post_training.pipeline import create_ptq_pipeline
from nncf.quantization.algorithms.weight_compression.algorithm import check_user_compression_configuration
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration
from nncf.quantization.telemetry_extractors import CompressionStartedWithCompressWeightsApi
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeWithAccuracyControlApi
from nncf.scopes import IgnoredScope
from nncf.telemetry.decorator import tracked_function
from nncf.telemetry.events import MODEL_BASED_CATEGORY

TTensor = TypeVar("TTensor")

BATCHWISE_STATISTICS_WARNING = (
    "For the particular model the batchwise statistics collection can lead to inaccurate statistics. "
    "If the accuracy degradation after compression is unsatisfactory, then "
    "the recommendation is to turn off batchwise statistics. If the results are still unsatisfactory, "
    "provide a dataloader with batch_size = 1 to the calibration dataset."
)


def warning_model_no_batchwise_support(
    graph: NNCFGraph,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters],
    model_type: ModelType,
    no_batchwise_support_metatypes: List[OperatorMetatype],
) -> None:
    """
    Logs when is_model_no_batchwise_support(...) returns True.

    :param graph: Model's NNCFGraph.
    :param advanced_quantization_parameters: AdvancedQuantizationParameters.
    :param model_type: Model type algorithm option.
    :param no_batchwise_support_metatypes: Metatypes having no batchwise statistics support.
    """
    if is_model_no_batchwise_support(
        graph, advanced_quantization_parameters, model_type, no_batchwise_support_metatypes
    ):
        nncf_logger.warning(BATCHWISE_STATISTICS_WARNING)


def is_model_no_batchwise_support(
    graph: NNCFGraph,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters],
    model_type: ModelType,
    no_batchwise_support_metatypes: List[OperatorMetatype],
) -> bool:
    """
    Returns True if batchwise statistics could lead to a significant accuracy drop.

    :param graph: Model's NNCFGraph.
    :param advanced_quantization_parameters: AdvancedQuantizationParameters.
    :param model_type: Model type algorithm option.
    :param no_batchwise_support_metatypes: Metatypes having no batchwise statistics support.
    """
    return bool(
        advanced_quantization_parameters
        and advanced_quantization_parameters.batchwise_statistics
        and (graph.get_nodes_by_metatypes(no_batchwise_support_metatypes) or model_type == ModelType.TRANSFORMER)
    )


def _update_advanced_quantization_parameters(
    advanced_parameters: Optional[AdvancedQuantizationParameters], calibration_dataset: Dataset
) -> AdvancedQuantizationParameters:
    """
    Updates AdvancedQuantizationParameters depending on batch_size.

    :param advanced_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :param calibration_dataset: A representative dataset for the
        calibration process.
    :return: Updated AdvancedQuantizationParameters.
    """
    batch_size = calibration_dataset.get_batch_size()
    if batch_size is not None and batch_size > 1:
        if advanced_parameters is None:
            advanced_parameters = AdvancedQuantizationParameters(batchwise_statistics=True)
        elif advanced_parameters.batchwise_statistics is None:
            advanced_parameters.batchwise_statistics = True
    return advanced_parameters


@api(canonical_alias="nncf.quantize")
@tracked_function(
    MODEL_BASED_CATEGORY,
    [
        CompressionStartedWithQuantizeApi(),
        "target_device",
        "preset",
    ],
)
def quantize(
    model: TModel,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> TModel:
    """
    Applies post-training quantization to the provided model.

    :param model: A model to be quantized.
    :type  model: TModel
    :param calibration_dataset: A representative dataset for the
        calibration process.
    :type  calibration_dataset: nncf.Dataset
    :param mode: Special quantization mode that specify different ways of the optimization.
    :type mode: Optional[nncf.QuantizationMode]
    :param preset: A preset controls the quantization mode (symmetric and asymmetric).
        It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
        Default value is None. In this case, `mixed` preset is used for `transformer`
        model type otherwise `performance`.
    :type  preset: nncf.QuantizationPreset
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :type  target_device: nncf.TargetDevice
    :param subset_size: Size of a subset to calculate activations statistics used for quantization.
        Must be positive.
    :param fast_bias_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :type  model_type: Optional[nncf.ModelType]
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :type  ignored_scope: Optional[nncf.IgnoredScope]
    :param advanced_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :return: The quantized model.
    :rtype: TModel
    """
    if subset_size < 1:
        raise nncf.ValidationError("Subset size must be positive.")

    advanced_parameters = _update_advanced_quantization_parameters(advanced_parameters, calibration_dataset)

    backend = get_backend(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )

    if backend == BackendType.ONNX:
        from nncf.onnx.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )

    if backend == BackendType.TENSORFLOW:
        from nncf.tensorflow.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )

    if backend == BackendType.TORCH:
        from nncf.torch.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )
    if backend == BackendType.TORCH_FX:
        from nncf.experimental.torch.fx.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model=model,
            calibration_dataset=calibration_dataset,
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )
    raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")


def wrap_validation_fn(validation_fn):
    """
    Wraps validation function to support case when it only returns metric value.

    :param validation_fn: Validation function to wrap.
    :return: Wrapped validation function.
    """

    def wrapper(*args, **kwargs):
        retval = validation_fn(*args, **kwargs)
        if isinstance(retval, tuple):
            return retval
        return retval, None

    return wrapper


@api(canonical_alias="nncf.quantize_with_accuracy_control")
@tracked_function(
    MODEL_BASED_CATEGORY,
    [
        CompressionStartedWithQuantizeWithAccuracyControlApi(),
        "target_device",
        "preset",
        "max_drop",
        "drop_type",
    ],
)
def quantize_with_accuracy_control(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    max_drop: float = 0.01,
    drop_type: DropType = DropType.ABSOLUTE,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters] = None,
    advanced_accuracy_restorer_parameters: Optional[AdvancedAccuracyRestorerParameters] = None,
) -> TModel:
    """
    Applies post-training quantization algorithm with accuracy control to provided model.

    :param model: A model to be quantized.
    :type model: TModel
    :param calibration_dataset: A representative dataset for the calibration process.
    :type calibration_dataset: nncf.Dataset
    :param validation_dataset: A dataset for the validation process.
    :type validation_dataset: nncf.Dataset
    :param validation_fn: A validation function to validate the model. It should take two arguments:
        - `model`: model to be validate.
        - `validation_dataset`: dataset that provides data items to
              validate the provided model.
        The function should return the value of the metric with the following meaning:
        A higher value corresponds to better performance of the model.
    :param max_drop: The maximum accuracy drop that should be achieved after the quantization.
    :param drop_type: The accuracy drop type, which determines how the maximum accuracy
        drop between the original model and the compressed model is calculated.
    :param preset: A preset controls the quantization mode (symmetric and asymmetric).
        It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
        Default value is None. In this case, `mixed` preset is used for `transformer`
        model type otherwise `performance`.
    :type preset: nncf.QuantizationPreset
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :type target_device: nncf.TargetDevice
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_bias_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :type model_type: nncf.ModelType
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :type ignored_scope: nncf.IgnoredScope
    :param advanced_quantization_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :param advanced_accuracy_restorer_parameters: Advanced parameters for fine-tuning
        the accuracy restorer algorithm.
    :type advanced_accuracy_restorer_parameters: Optional[AdvancedAccuracyRestorerParameters]
    :return: The quantized model.
    :rtype: TModel
    """
    advanced_quantization_parameters = _update_advanced_quantization_parameters(
        advanced_quantization_parameters, calibration_dataset
    )

    backend = get_backend(model)

    validation_fn = wrap_validation_fn(validation_fn)

    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import quantize_with_accuracy_control_impl

        return quantize_with_accuracy_control_impl(
            model,
            calibration_dataset,
            validation_dataset,
            validation_fn,
            max_drop,
            drop_type,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_quantization_parameters,
            advanced_accuracy_restorer_parameters,
        )
    if backend == BackendType.ONNX:
        from nncf.onnx.quantization.quantize_model import quantize_with_accuracy_control_impl

        return quantize_with_accuracy_control_impl(
            model,
            calibration_dataset,
            validation_dataset,
            validation_fn,
            max_drop,
            drop_type,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_quantization_parameters,
            advanced_accuracy_restorer_parameters,
        )

    raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")


@api(canonical_alias="nncf.compress_weights")
@tracked_function(
    MODEL_BASED_CATEGORY,
    [
        CompressionStartedWithCompressWeightsApi(),
        "mode",
        "awq",
        "scale_estimation",
        "gptq",
        "lora_correction",
        "backup_mode",
    ],
)
def compress_weights(
    model: TModel,
    mode=CompressWeightsMode.INT8_ASYM,
    ratio: Optional[float] = None,
    group_size: Optional[int] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    all_layers: Optional[bool] = None,
    dataset: Optional[Dataset] = None,
    sensitivity_metric: Optional[SensitivityMetric] = None,
    *,
    subset_size: int = 128,
    awq: Optional[bool] = None,
    scale_estimation: Optional[bool] = None,
    gptq: Optional[bool] = None,
    lora_correction: Optional[bool] = None,
    backup_mode: Optional[BackupMode] = None,
    advanced_parameters: Optional[AdvancedCompressionParameters] = None,
) -> TModel:
    """
    Compress model weights.

    :param model: A model to be compressed.
    :type model: TModel
    :param mode: Defines a mode for weight compression.
        INT8_SYM stands for 8-bit integer symmetric quantization of all weights without zero point.
        INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
            Weights are quantized to a primary precision symmetrically without zero point.
            All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
            by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
            criteria and the given ratio.
        INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        E2M1 is the same as INT4_SYM mode, but primary precision is E2M1 data type without zero point.
    :type mode: nncf.CompressWeightsMode
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8_ASYM).
    :type ratio: float
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    :type group_size: int
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :type ignored_scope: nncf.IgnoredScope
    :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
        precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
    :type all_layers: bool
    :param dataset: Dataset used for assigning different quantization precision by finding outliers in activations.
    :type dataset: nncf.Dataset
    :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
        preserve the accuracy of the model, the more sensitive layers receives a higher precision.
    :type sensitivity_metric: nncf.SensitivityMetric
    :param subset_size: Number of data samples to calculate activation statistics used for assigning different
        quantization precision. Defaults to 128.
    :type subset_size: int
    :param awq: Indicates whether use AWQ weights correction.
    :type awq: bool
    :param scale_estimation: Indicates whether a scale estimation algorithm is used that minimizes the L2 error
        between the original and compressed layers.
    :type scale_estimation: bool
    :param gptq: Indicates whether to use GPTQ algorithm.
    :type gptq: bool
    :param lora_correction: Indicates whether to use Lora Correction algorithm.
    :type lora_correction: bool
    :param backup_mode: Defines a backup mode for mixed-precision weight compression.
        NONE stands for original floating-point precision of the model weights.
            In this mode, weights are retained in their original precision without any quantization.
        INT8_SYM stands for 8-bit integer symmetric quantization without zero point.
        INT8_ASYM stands for 8-bit integer asymmetric quantization with a typical non-fixed zero point.
    :type backup_mode: nncf.BackupMode
    :param advanced_parameters: Advanced parameters for compression algorithms.
    :type advanced_parameters: nncf.AdvancedCompressionParameters
    :return: The non-trainable model with compressed weights.
    """
    if mode == CompressWeightsMode.INT8:
        warning_deprecated(
            "`CompressWeightsMode.INT8` is deprecated. Please, use `CompressWeightsMode.INT8_ASYM` as value instead."
        )
        mode = CompressWeightsMode.INT8_ASYM

    backend = get_backend(model)
    compression_weights_impl = None

    if backend == BackendType.TORCH:
        from nncf.torch.model_creation import is_wrapped_model
        from nncf.torch.model_creation import wrap_model
        from nncf.torch.quantization.quantize_model import compress_weights_impl as pt_compression_weights_impl

        if mode in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]:
            raise nncf.ParameterNotSupportedError(
                "Torch backend does not support NF4 and E2M1 modes for weight compression."
            )

        options = {
            "awq": awq,
            "scale_estimation": scale_estimation,
            "gptq": gptq,
            "lora_correction": lora_correction,
        }
        unsupported_options = [name for name, value in options.items() if value is not None]
        if unsupported_options:
            raise nncf.ParameterNotSupportedError(
                f"Torch backend does not support {', '.join(unsupported_options)} option(s). Set them to None."
            )

        if sensitivity_metric not in [None, SensitivityMetric.WEIGHT_QUANTIZATION_ERROR]:
            raise nncf.ParameterNotSupportedError(
                "Torch backend only supports data-free sensitivity metric. "
                "Set None or SensitivityMetric.WEIGHT_QUANTIZATION_ERROR."
            )

        if advanced_parameters and advanced_parameters.statistics_path:
            raise nncf.ParameterNotSupportedError("Torch does not support statistics caching.")

        if is_wrapped_model(model):
            if not model.nncf.trace_parameters:
                raise nncf.ValidationError(
                    "Tracing capabilities with tracing parameters are required in the PyTorch model "
                    "for nncf.compress_weights(). Please wrap the model using "
                    "nncf.torch.wrap_model(model, example_input, trace_parameters=True) before calling "
                    "nncf.compress_weights()."
                )
        elif dataset is None:
            raise nncf.ValidationError("Please provide a dataset of at least one element for PyTorch model tracing.")
        else:
            example_input = next(iter(dataset.get_inference_data()))
            model = wrap_model(model, example_input=example_input, trace_parameters=True)
        dataset = None
        compression_weights_impl = pt_compression_weights_impl

    if backend == BackendType.TORCH_FX:
        from nncf.experimental.torch.fx.quantization.quantize_model import (
            compress_weights_impl as fx_compression_weights_impl,
        )

        if mode in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]:
            raise nncf.ParameterNotSupportedError(
                "Torch backend does not support NF4 and E2M1 modes for weight compression."
            )

        options = {
            "awq": awq,
            "scale_estimation": scale_estimation,
            "gptq": gptq,
            "lora_correction": lora_correction,
        }
        unsupported_options = [name for name, value in options.items() if value is not None]
        if unsupported_options:
            raise nncf.ParameterNotSupportedError(
                f"TorchFX backend does not support {', '.join(unsupported_options)} option(s). Set them to None."
            )

        if sensitivity_metric not in [None, SensitivityMetric.WEIGHT_QUANTIZATION_ERROR]:
            raise nncf.ParameterNotSupportedError(
                "TorchFX backend only supports data-free sensitivity metric. "
                "Set None or SensitivityMetric.WEIGHT_QUANTIZATION_ERROR."
            )

        if dataset:
            raise nncf.ParameterNotSupportedError(
                "TorchFX only supports data-free weights compression," "Set the 'dataset' option to None"
            )
        if advanced_parameters and advanced_parameters.statistics_path:
            raise nncf.ParameterNotSupportedError("TorchFX does not supports statistics caching.")
        compression_weights_impl = fx_compression_weights_impl

    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import compress_weights_impl as ov_compress_weights_impl

        if any((awq, scale_estimation, gptq, lora_correction)) and (
            dataset is None or mode == CompressWeightsMode.E2M1
        ):
            raise nncf.ParameterNotSupportedError(
                "Scale estimation, AWQ, GPTQ or Lora Correction algorithm is defined, "
                "but dataset is None or mode is E2M1."
            )

        if gptq and lora_correction:
            raise nncf.ValidationError(
                "Simultaneous use of Lora correction and GPTQ algorithms is not supported. Select one of them."
            )

        compression_weights_impl = ov_compress_weights_impl
    check_user_compression_configuration(
        mode,
        subset_size,
        dataset,
        ratio,
        group_size,
        all_layers,
        awq,
        scale_estimation,
        gptq,
        lora_correction,
        ignored_scope,
        sensitivity_metric,
        backup_mode,
        advanced_parameters,
    )
    weight_compression_configuration = get_weight_compression_configuration(
        mode,
        dataset,
        ratio,
        group_size,
        all_layers,
        awq,
        scale_estimation,
        gptq,
        lora_correction,
        ignored_scope,
        sensitivity_metric,
        backup_mode,
        advanced_parameters,
    )

    if compression_weights_impl is None:
        raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")

    return compression_weights_impl(
        model=model,
        dataset=dataset,
        subset_size=subset_size,
        **weight_compression_configuration,
    )


def quantize_with_tune_hyperparams(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    initial_metric_results: MetricResults,
    quantized_metric_results: MetricResults,
    tuner_subset_size: int = 300,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> TModel:
    """
    Applies post-training quantization algorithm with tune hyperparameters to provided model.

    :param model: A model to be quantized.
    :param calibration_dataset: A representative dataset for the calibration process.
    :param validation_dataset: : A dataset for the validation process.
    :param validation_fn: A validation function to validate the model.
    :param initial_metric_results: Initial metric results.
    :param quantized_metric_results: Quantized metric results.
    :param tuner_subset_size: Tuner subset size.
    :param preset: A preset controls the quantization mode (symmetric and asymmetric).
        It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
        Default value is None. In this case, `mixed` preset is used for `transformer`
        model type otherwise `performance`.
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
    :param advanced_quantization_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :return: The quantized model.
    """
    init_quantization_params = {
        "preset": preset,
        "target_device": target_device,
        "subset_size": subset_size,
        "fast_bias_correction": fast_bias_correction,
        "model_type": model_type,
        "ignored_scope": ignored_scope,
        "advanced_parameters": advanced_quantization_parameters,
    }

    backend = get_backend(model)
    param_grids = get_quantization_param_grids(create_ptq_pipeline(**init_quantization_params), backend)

    hyperparameter_tuner = HyperparameterTuner(
        create_ptq_pipeline,
        init_quantization_params,
        param_grids,
        calibration_dataset,
        validation_fn,
        tuner_subset_size,
        initial_metric_results,
        quantized_metric_results,
    )

    quantized_model = hyperparameter_tuner.apply(model, validation_dataset)

    return quantized_model
