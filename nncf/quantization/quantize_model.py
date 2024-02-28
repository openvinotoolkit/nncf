# Copyright (c) 2024 Intel Corporation
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
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.parameters import CompressWeightsMode
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import HyperparameterTuner
from nncf.quantization.algorithms.hyperparameter_tuner.param_grid import get_quantization_param_grids
from nncf.quantization.algorithms.post_training.pipeline import create_ptq_pipeline
from nncf.scopes import IgnoredScope

TTensor = TypeVar("TTensor")


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
            advanced_parameters = AdvancedQuantizationParameters(statistics_per_sample=True)
        elif advanced_parameters.statistics_per_sample is None:
            advanced_parameters.statistics_per_sample = True
    return advanced_parameters


@api(canonical_alias="nncf.quantize")
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
        raise ValueError("Subset size must be positive.")

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

    raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")


@api(canonical_alias="nncf.quantize_with_accuracy_control")
def quantize_with_accuracy_control(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], float],
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
    subset_size: Optional[int] = 128,
    awq: Optional[bool] = None,
) -> TModel:
    """
    Compress model weights.

    :param model: A model to be compressed.
    :param mode: Defines a mode for weight compression.
        INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
        INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
            Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
            All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
            by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
            criteria and the given ratio.
        INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8_ASYM).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
        precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
    :param dataset: Dataset used for assigning different quantization precision by finding outliers in activations.
    :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
        preserve the accuracy of the model, the more sensitive layers receives a higher precision.
    :param subset_size: Number of data samples to calculate activation statistics used for assigning different
        quantization precision. Defaults to 128.
    :param awq: Indicates whether use AWQ weights correction.
    :return: The non-trainable model with compressed weights.
    """
    if mode == CompressWeightsMode.INT8:
        warning_deprecated(
            "`CompressWeightsMode.INT8` is deprecated." "Please, use `CompressWeightsMode.INT8_ASYM` as value instead."
        )
        mode = CompressWeightsMode.INT8_ASYM

    backend = get_backend(model)
    compression_weights_impl = None

    if backend == BackendType.TORCH:
        from nncf.torch.model_creation import is_wrapped_model
        from nncf.torch.model_creation import wrap_model
        from nncf.torch.quantization.quantize_model import compress_weights_impl as pt_compression_weights_impl

        if mode not in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]:
            raise AttributeError(
                "Torch backend supports only INT8_ASYM, INT8_SYM modes for weight compression, "
                f"but given {mode.value} mode."
            )

        if awq is True:
            raise AttributeError("Torch backend doesn`t supports AWQ algorithm, but awq=True is specified.")

        if is_wrapped_model(model):
            if not model.nncf.trace_parameters:
                raise ValueError(
                    "Tracing capabilities with tracing parameters are required in the PyTorch model "
                    "for nncf.compress_weights(). Please wrap the model using "
                    "nncf.torch.wrap_model(model, example_input, trace_parameters=True) before calling "
                    "nncf.compress_weights()."
                )
        elif dataset is None:
            raise AttributeError("Please provide a dataset of at least one element for PyTorch model tracing.")
        else:
            example_input = next(iter(dataset.get_inference_data()))
            model = wrap_model(model, example_input=example_input, trace_parameters=True)
            dataset = None
        compression_weights_impl = pt_compression_weights_impl

    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import compress_weights_impl as ov_compress_weights_impl

        compression_weights_impl = ov_compress_weights_impl

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]:
        if ratio is None:
            ratio = 1
        if group_size is None:
            group_size = -1
        if ratio != 1 or group_size != -1:
            raise AttributeError(
                "INT8 mode assumes per-channel quantization of all layers in 8 bit. "
                "Default values of `ratio` (1) and `group_size` (-1) parameters can not be overridden"
            )
        options = [all_layers, sensitivity_metric, dataset, awq]
        if any(option is not None for option in options):
            raise AttributeError(
                "INT8 modes do not support `all_layers`, `sensitivity_metric`, `awq` and `dataset` options. "
                "Set them to None."
            )

    if ratio is None:
        ratio = 1
    if group_size is None:
        group_size = 128
    if all_layers is None:
        all_layers = False
    if awq is None:
        awq = False
    if ignored_scope is None:
        ignored_scope = IgnoredScope()
    if sensitivity_metric is None:
        sensitivity_metric = (
            SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
            if dataset is None
            else SensitivityMetric.MAX_ACTIVATION_VARIANCE
        )
    if ratio != 1 and dataset is None and sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR:
        raise AttributeError(
            f"Mixed precision selection based on the given sensitivity metric={sensitivity_metric.value} requires "
            "a dataset, but it's not provided."
        )
    if ratio < 0 or ratio > 1:
        raise ValueError(f"The ratio should be between 0 and 1, but ratio={ratio} is specified.")
    if subset_size is None or subset_size <= 0:
        raise ValueError(f"The subset_size value should be positive, but subset_size={subset_size} is given.")

    if compression_weights_impl is None:
        raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")

    return compression_weights_impl(
        model, dataset, mode, ratio, group_size, ignored_scope, all_layers, sensitivity_metric, awq, subset_size
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
