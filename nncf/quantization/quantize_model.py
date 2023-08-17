# Copyright (c) 2023 Intel Corporation
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

from nncf.api.compression import TModel
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import HyperparameterTuner
from nncf.quantization.algorithms.hyperparameter_tuner.param_grid import get_quantization_param_grid
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope

TTensor = TypeVar("TTensor")


@api(canonical_alias="nncf.quantize")
def quantize(
    model: TModel,
    calibration_dataset: Dataset,
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
    :param preset: A preset that controls the quantization mode
        (symmetric and asymmetric). It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric
          quantization of activations.
    :type  preset: nncf.QuantizationPreset
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :type  target_device: nncf.TargetDevice
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
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
    backend = get_backend(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model,
            calibration_dataset,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_parameters,
        )

    if backend == BackendType.ONNX:
        from nncf.onnx.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model,
            calibration_dataset,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_parameters,
        )

    if backend == BackendType.TENSORFLOW:
        from nncf.tensorflow.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model,
            calibration_dataset,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_parameters,
        )

    if backend == BackendType.TORCH:
        from nncf.torch.quantization.quantize_model import quantize_impl

        return quantize_impl(
            model,
            calibration_dataset,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_parameters,
        )

    raise RuntimeError(f"Unsupported type of backend: {backend}")


@api(canonical_alias="nncf.quantize_with_accuracy_control")
def quantize_with_accuracy_control(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], float],
    max_drop: float = 0.01,
    drop_type: DropType = DropType.ABSOLUTE,
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
    :param preset: A preset that controls the quantization mode.
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

    raise RuntimeError(f"Unsupported type of backend: {backend}")


@api(canonical_alias="nncf.compress_weights")
def compress_weights(model: TModel) -> TModel:
    """
    Compress model weights.

    :param model: A model to be compressed.
    :return: The non-trainable model with compressed weights.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        import nncf.torch

        return nncf.torch.compress_weights(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.quantize_model import compress_weights
        return compress_weights(model)

    raise RuntimeError(f"Unsupported type of backend: {backend}")


def quantize_with_tune_hyperparams(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    initial_metric_results: MetricResults,
    quantized_metric_results: MetricResults,
    tuner_subset_size: int = 300,
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
    :param preset: A preset that controls the quantization mode.
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

    quantization_param_grid = get_quantization_param_grid()

    hyperparameter_tuner = HyperparameterTuner(
        PostTrainingQuantization,
        init_quantization_params,
        quantization_param_grid,
        calibration_dataset,
        validation_fn,
        tuner_subset_size,
        initial_metric_results,
        quantized_metric_results,
    )

    quantized_model = hyperparameter_tuner.apply(model, validation_dataset)

    return quantized_model
