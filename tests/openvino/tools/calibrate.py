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

import json
import multiprocessing
import os
from argparse import ArgumentParser
from collections import OrderedDict
from collections import defaultdict
from dataclasses import asdict
from enum import Enum
from itertools import islice
from typing import Iterable, Optional, TypeVar

import numpy as np
import openvino.runtime as ov
from openvino.runtime import Dimension
from openvino.runtime import PartialShape
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
from openvino.tools.pot.configs.config import Config

import nncf
from nncf.common.logging.logger import set_log_file
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.openvino.quantization.quantize_model import (
    quantize_with_accuracy_control as pot_quantize_with_native_accuracy_control,
)
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AggregatorType
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")

MAP_POT_NNCF_ALGORITHMS = {
    "DefaultQuantization": "quantize",
    "AccuracyAwareQuantization": "quantize_with_accuracy_control",
}

_default_context = None


def parse_args():
    """
    Parses command line arguments.

    :return: A dict with command-line arguments
    """
    parser = ArgumentParser(description="NNCF OpenVINO Benchmarking Tool", allow_abbrev=False)

    parser.add_argument("-c", "--config", help="Path to a config file with optimization parameters (POT format).")

    parser.add_argument(
        "--output-dir", type=str, default="./results", help="The directory where models are saved. Default: ./results"
    )

    parser.add_argument("--impl", help="NNCF OpenVINO backend implementation.", choices=["pot", "native"], default=None)

    return parser.parse_args()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(
            o, (TargetDevice, ModelType, QuantizationPreset, OverflowFix, StatisticsType, AggregatorType, DropType)
        ):
            return o.value
        if isinstance(o, (IgnoredScope, AdvancedQuantizationParameters, AdvancedAccuracyRestorerParameters)):
            return asdict(o)
        raise TypeError(f"Object of type {o.__class__.__name__} " f"is not JSON serializable")


class ACValidationFunction:
    """
    Implementation of a validation function using the Accuracy Checker.
    """

    def __init__(self, model_evaluator: ModelEvaluator, metric_name: str, requests_number: Optional[int] = None):
        """
        :param model_evaluator: Model Evaluator.
        :param metric_name: Name of a metric.
        :param requests_number: A number of infer requests. If it is `None`,
            the count will be selected automatically.
        """
        self._model_evaluator = model_evaluator
        self._metric_name = metric_name
        self._requests_number = requests_number

    def __call__(self, compiled_model: ov.CompiledModel, indices: Optional[Iterable[int]] = None) -> float:
        """
        Calculates metrics for the provided model.

        :param compiled_model: A compiled model to validate.
        :param indices: The zero-based indices of data items
            that should be selected from the whole dataset.
        :return: Calculated metrics.
        """
        self._model_evaluator.launcher.exec_network = compiled_model
        self._model_evaluator.launcher.infer_request = compiled_model.create_infer_request()

        if indices:
            indices = list(indices)

        kwargs = {
            "subset": indices,
            "check_progress": False,
            "dataset_tag": "",
            "calculate_metrics": True,
        }

        if self._requests_number == 1:
            self._model_evaluator.process_dataset(**kwargs)
        else:
            self._set_requests_number(kwargs, self._requests_number)
            self._model_evaluator.process_dataset_async(**kwargs)

        # Calculate metrics
        metrics = OrderedDict()
        for metric in self._model_evaluator.compute_metrics(print_results=False):
            sign = 1.0
            if metric.meta.get("target", "higher-better") == "higher-worse":
                sign = -1.0

            if metric.meta.get("calculate_mean", True):
                metric_value = np.mean(metric.evaluated_value)
            else:
                metric_value = metric.evaluated_value[0]

            metrics[metric.name] = sign * metric_value

        self._model_evaluator.reset()

        return metrics[self._metric_name]

    @staticmethod
    def _set_requests_number(params, requests_number):
        if requests_number:
            params["nreq"] = np.clip(requests_number, 1, multiprocessing.cpu_count())
            if params["nreq"] != requests_number:
                print(
                    "Number of requests {} is out of range [1, {}]. Will be used {}.".format(
                        requests_number, multiprocessing.cpu_count(), params["nreq"]
                    )
                )


def dump_to_json(path, value, keys):
    dump_value = {}
    for k in keys:
        dump_value[k] = value[k]

    with open(path, "w", encoding="utf8") as f:
        json.dump(dump_value, f, cls=CustomJSONEncoder)


class BlockedGroups(Enum):
    activations_min_range = "activations_min_range"
    activations_max_range = "activations_max_range"
    weights_min_range = "weights_min_range"
    weights_max_range = "weights_max_range"


class ParameterNames(Enum):
    advanced_parameters = "advanced_parameters"


class AlgorithmParametersContext:
    def __init__(self, param_name_map):
        self.params = {}
        self.blocked_params = {}
        for group in BlockedGroups:
            self.blocked_params[group] = []
        self.param_name_map = param_name_map
        self._save_context = None

    def __enter__(self):
        self._save_context = get_algorithm_parameters_context()
        set_algorithm_parameters_context(self)
        return self

    def __exit__(self, *args):
        set_algorithm_parameters_context(self._save_context)
        self._save_context = None


def get_algorithm_parameters_context():
    return _default_context


def set_algorithm_parameters_context(ctx):
    global _default_context
    _default_context = ctx


def map_target_device(target_device):
    target_device = target_device.upper()
    if target_device not in [t.value for t in TargetDevice]:
        raise ValueError(f"{target_device} target device is not supported")
    return {"target_device": TargetDevice(target_device)}


def map_model_type(model_type):
    model_type = model_type.lower()
    if model_type not in [m.value for m in ModelType]:
        raise ValueError(f"{model_type} model type is not supported")
    return {"model_type": ModelType(model_type)}


def map_drop_type(drop_type):
    drop_type = drop_type.lower()
    if drop_type not in [m.value for m in DropType]:
        raise ValueError(f"{drop_type} drop type is not supported")
    return {"drop_type": DropType(drop_type)}


def map_ignored_scope(ignored):
    if ignored.get("skip_model") is not None:
        raise ValueError("skip_model attribute in the ignored tag is not supported")

    operations = ignored.get("operations")
    ignored_operations = []
    if operations is not None:
        for op in operations:
            if op.get("attributes") is not None:
                raise ValueError('"attributes" in the ignored operations ' "are not supported")
            ignored_operations.append(op["type"])
    return {"ignored_scope": IgnoredScope(names=ignored.get("scope"), types=ignored_operations)}


def map_preset(preset):
    preset = preset.lower()
    if preset not in [p.value for p in QuantizationPreset]:
        raise ValueError(f"{preset} preset is not supported")
    return {"preset": QuantizationPreset(preset)}


def map_inplace_statistics(inplace_statistics):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())
    advanced_parameters.inplace_statistics = inplace_statistics
    return {advanced_parameter_name: advanced_parameters}


def update_statistics_collector_parameters(
    stat_collector_params, blocked_params, pot_config, is_global_range_estimator
):
    granularity = pot_config.get("granularity")
    if granularity is not None:
        raise ValueError('"granularity" parameter in the range estimator is not supported')

    stat_collector_names = ["statistics_type", "aggregator_type", "clipping_value", "quantile_outlier_prob"]
    stat_collector_types = [StatisticsType, AggregatorType, float, float]
    pot_names = ["type", "aggregator", "clipping_value", "outlier_prob"]

    for stat_collector_name, pot_name, stat_collector_type in zip(
        stat_collector_names, pot_names, stat_collector_types
    ):
        if stat_collector_name in blocked_params and is_global_range_estimator:
            continue

        pot_value = pot_config.get(pot_name)
        if pot_value is not None:
            if not is_global_range_estimator:
                blocked_params.append(stat_collector_name)
            setattr(stat_collector_params, stat_collector_name, stat_collector_type(pot_value))


def update_range_estimator_parameters(
    range_estimator_params, pot_config, min_blocked_params, max_blocked_params, is_global_range_estimator
):
    preset = pot_config.get("preset")
    if preset is not None:
        raise ValueError('"preset" parameter in the range estimator is not supported')

    min_config = pot_config.get("min")
    max_config = pot_config.get("max")
    if min_config is None and max_config is None:
        return

    if min_config is not None:
        update_statistics_collector_parameters(
            range_estimator_params.min, min_blocked_params, min_config, is_global_range_estimator
        )
    if max_config is not None:
        update_statistics_collector_parameters(
            range_estimator_params.max, max_blocked_params, max_config, is_global_range_estimator
        )


def map_range_estmator(range_estimator):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())

    update_range_estimator_parameters(
        advanced_parameters.activations_range_estimator_params,
        range_estimator,
        ctx.blocked_params[BlockedGroups.activations_min_range],
        ctx.blocked_params[BlockedGroups.activations_max_range],
        True,
    )
    update_range_estimator_parameters(
        advanced_parameters.weights_range_estimator_params,
        range_estimator,
        ctx.blocked_params[BlockedGroups.weights_min_range],
        ctx.blocked_params[BlockedGroups.weights_max_range],
        True,
    )
    return {advanced_parameter_name: advanced_parameters}


def update_quantization_parameters(quantization_params, pot_config):
    level_low = pot_config.get("level_low")
    if level_low is not None:
        raise ValueError('"level_low" parameter is not supported')
    level_high = pot_config.get("level_high")
    if level_high is not None:
        raise ValueError('"level_high" parameter is not supported')
    num_bits = pot_config.get("bits")
    if num_bits is not None:
        quantization_params.num_bits = num_bits
    mode = pot_config.get("mode")
    if mode is not None:
        if mode == "symmetric":
            quantization_params.mode = QuantizationMode.SYMMETRIC
        elif mode == "asymmetric":
            quantization_params.mode = QuantizationMode.ASYMMETRIC
        else:
            raise ValueError(f"mode = {mode} is not supported")
    granularity = pot_config.get("granularity")
    if granularity is not None:
        if granularity == "perchannel":
            quantization_params.per_channel = True
        elif mode == "pertensor":
            quantization_params.per_channel = False
        else:
            raise ValueError(f"granularity = {granularity} is not supported")


def map_weights(weights):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())

    update_quantization_parameters(advanced_parameters.weights_quantization_params, weights)

    range_estimator = weights.get("range_estimator")
    if range_estimator is not None:
        update_range_estimator_parameters(
            advanced_parameters.weights_range_estimator_params,
            range_estimator,
            ctx.blocked_params[BlockedGroups.weights_min_range],
            ctx.blocked_params[BlockedGroups.weights_max_range],
            False,
        )
    return {advanced_parameter_name: advanced_parameters}


def map_activations(activations):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())

    update_quantization_parameters(advanced_parameters.weights_quantization_params, activations)

    range_estimator = activations.get("range_estimator")
    if range_estimator is not None:
        update_range_estimator_parameters(
            advanced_parameters.activations_range_estimator_params,
            range_estimator,
            ctx.blocked_params[BlockedGroups.activations_min_range],
            ctx.blocked_params[BlockedGroups.activations_max_range],
            False,
        )
    return {advanced_parameter_name: advanced_parameters}


def map_saturation_fix(saturation_fix):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())
    if saturation_fix == "no":
        advanced_parameters.overflow_fix = OverflowFix.DISABLE
    elif saturation_fix == "first_layer":
        advanced_parameters.overflow_fix = OverflowFix.FIRST_LAYER
    elif saturation_fix == "all":
        advanced_parameters.overflow_fix = OverflowFix.ENABLE
    else:
        ValueError(f"saturation_fix = {saturation_fix} is not supported")
    return {advanced_parameter_name: advanced_parameters}


def map_apply_for_all_nodes(apply_for_all_nodes):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())
    advanced_parameters.bias_correction_params.apply_for_all_nodes = apply_for_all_nodes
    return {advanced_parameter_name: advanced_parameters}


def map_threshold(threshold):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())
    advanced_parameters.bias_correction_params.threshold = threshold
    return {advanced_parameter_name: advanced_parameters}


def map_max_iter_num(max_iter_num):
    ctx = get_algorithm_parameters_context()
    advanced_parameters = ctx.params.get("advanced_accuracy_restorer_parameters", AdvancedAccuracyRestorerParameters())
    advanced_parameters.max_num_iterations = max_iter_num
    return {"advanced_accuracy_restorer_parameters": advanced_parameters}


def map_ranking_subset_size(ranking_subset_size):
    ctx = get_algorithm_parameters_context()
    advanced_parameters = ctx.params.get("advanced_accuracy_restorer_parameters", AdvancedAccuracyRestorerParameters())
    advanced_parameters.ranking_subset_size = ranking_subset_size
    return {"advanced_accuracy_restorer_parameters": advanced_parameters}


def map_tune_hyperparams(tune_hyperparams):
    ctx = get_algorithm_parameters_context()
    advanced_parameters = ctx.params.get("advanced_accuracy_restorer_parameters", AdvancedAccuracyRestorerParameters())
    advanced_parameters.tune_hyperparams = tune_hyperparams
    return {"advanced_accuracy_restorer_parameters": advanced_parameters}


def map_convert_to_mixed_preset(convert_to_mixed_preset):
    ctx = get_algorithm_parameters_context()
    advanced_parameters = ctx.params.get("advanced_accuracy_restorer_parameters", AdvancedAccuracyRestorerParameters())
    advanced_parameters.convert_to_mixed_preset = convert_to_mixed_preset
    return {"advanced_accuracy_restorer_parameters": advanced_parameters}


def create_parameters_for_algorithm(
    pot_parameters, supported_parameters, default_parameters, ignored_parameters, param_name_map
):
    with AlgorithmParametersContext(param_name_map) as ctx:
        for name in pot_parameters:
            if (
                pot_parameters[name] is None
                or name in ignored_parameters
                or (name in default_parameters and pot_parameters[name] == default_parameters[name])
            ):
                continue
            if name in supported_parameters:
                kwarg = supported_parameters[name](pot_parameters[name])
                if kwarg is not None:
                    ctx.params.update(kwarg)
            else:
                raise ValueError(f"{name} parameter is not supported")

        return ctx.params


def get_pot_quantization_parameters_mapping():
    supported_parameters = {
        "target_device": map_target_device,
        "model_type": map_model_type,
        "ignored": map_ignored_scope,
        "preset": map_preset,
        "stat_subset_size": lambda x: {"subset_size": x},
        "use_fast_bias": lambda x: {"fast_bias_correction": x},
        "inplace_statistics": map_inplace_statistics,
        "range_estimator": map_range_estmator,
        "weights": map_weights,
        "activations": map_activations,
        "saturation_fix": map_saturation_fix,
        "apply_for_all_nodes": map_apply_for_all_nodes,
        "threshold": map_threshold,
    }

    default_parameters = {"use_layerwise_tuning": False}

    ignored_parameters = [
        "dump_intermediate_model",
        "num_samples_for_tuning",
        "batch_size",
        "optimizer",
        "loss",
        "tuning_iterations",
        "random_seed",
        "use_ranking_subset",
        "calibration_indices_pool",
        "calculate_grads_on_loss_increase_only",
        "weight_decay",
        "seed",
    ]

    return supported_parameters, default_parameters, ignored_parameters


def map_quantization_parameters(pot_parameters):
    supported_parameters, default_parameters, ignored_parameters = get_pot_quantization_parameters_mapping()

    param_name_map = {ParameterNames.advanced_parameters: "advanced_parameters"}

    result = create_parameters_for_algorithm(
        pot_parameters, supported_parameters, default_parameters, ignored_parameters, param_name_map
    )

    return result


def map_quantize_with_accuracy_control_parameters(pot_parameters):
    supported_parameters, default_parameters, ignored_parameters = get_pot_quantization_parameters_mapping()

    supported_parameters.update(
        {
            "maximal_drop": lambda x: {"max_drop": x},
            "max_iter_num": map_max_iter_num,
            "ranking_subset_size": map_ranking_subset_size,
            "tune_hyperparams": map_tune_hyperparams,
            "convert_to_mixed_preset": map_convert_to_mixed_preset,
            "drop_type": map_drop_type,
        }
    )

    default_parameters.update(
        {
            "use_prev_if_drop_increase": True,
            "base_algorithm": "DefaultQuantization",
            "annotation_free": False,
        }
    )

    ignored_parameters.extend(
        [
            "annotation_conf_threshold",
            "metric_subset_ratio",
        ]
    )

    param_name_map = {ParameterNames.advanced_parameters: "advanced_quantization_parameters"}

    result = create_parameters_for_algorithm(
        pot_parameters, supported_parameters, default_parameters, ignored_parameters, param_name_map
    )

    return result


def map_paramaters(pot_algo_name, nncf_algo_name, pot_parameters):
    if pot_algo_name == "DefaultQuantization" and nncf_algo_name == "quantize":
        return map_quantization_parameters(pot_parameters)
    if pot_algo_name == "AccuracyAwareQuantization" and nncf_algo_name == "quantize_with_accuracy_control":
        return map_quantize_with_accuracy_control_parameters(pot_parameters)
    raise ValueError(f"Mapping POT {pot_algo_name} parameters to NNCF " f"{nncf_algo_name} parameters is not supported")


def get_model_paths(model_config):
    if model_config.cascade:
        raise ValueError("Cascade models are not supported yet.")
    return model_config.model, model_config.weights


def get_accuracy_checker_config(engine_config):
    if engine_config.type != "accuracy_checker":
        raise ValueError(f"Engine type {engine_config.type} is not supported.")
    return engine_config


def get_nncf_algorithms_config(compression_config):
    nncf_algorithms = []
    for pot_algo in compression_config.algorithms:
        if pot_algo.name not in MAP_POT_NNCF_ALGORITHMS:
            raise ValueError(f"Algorithm {pot_algo.name} is not supported.")

        nncf_algo_name = MAP_POT_NNCF_ALGORITHMS[pot_algo.name]
        nncf_algorithms.append(
            {"name": nncf_algo_name, "parameters": map_paramaters(pot_algo.name, nncf_algo_name, pot_algo.params)}
        )
    return nncf_algorithms


def get_allow_reshape_input(accuracy_checker_config) -> bool:
    for model_config in accuracy_checker_config["models"]:
        for launcher_config in model_config["launchers"]:
            if "allow_reshape_input" in launcher_config:
                return launcher_config["allow_reshape_input"]
    return False


# pylint:disable=too-many-branches
def maybe_reshape_model(model, dataset, subset_size, input_to_tensor_name):
    dataset_inputs_shapes = defaultdict(set)
    for input_dict in islice(dataset.get_inference_data(), subset_size):
        for name, tensor in input_dict.items():
            dataset_inputs_shapes[name].add(tuple(tensor.shape))

    model_inputs_shapes = {}
    for input_output in model.inputs:
        input_node = input_output.get_node()
        model_inputs_shapes[input_to_tensor_name[input_node.friendly_name]] = tuple(input_node.partial_shape)

    if len(dataset_inputs_shapes) != len(model_inputs_shapes):
        raise RuntimeError(
            f"Model inputs: {list(model_inputs_shapes.keys())}"
            f" and dataset inputs {list(dataset_inputs_shapes.keys())} are not compatible"
        )

    for name in model_inputs_shapes:
        if name not in dataset_inputs_shapes:
            raise RuntimeError(
                f"Model input {name} is not present in dataset inputs: {list(dataset_inputs_shapes.keys())}"
            )

    dynamic_dims = defaultdict(list)
    reshaped_static_dims = defaultdict(list)
    for name, shapes in dataset_inputs_shapes.items():
        shapes = list(shapes)
        if len(set(len(shape) for shape in shapes)) != 1 or len(model_inputs_shapes[name]) != len(shapes[0]):
            raise RuntimeError("calibrate.py does not support dataset with dynamic ranks")

        for idx in range(len(shapes[0])):
            if len(shapes) == 1:
                model_dim = model_inputs_shapes[name][idx]
                if model_dim.is_static and model_dim.get_length() != shapes[0][idx]:
                    reshaped_static_dims[name].append(idx)

            elif any(shapes[0][idx] != shape[idx] for shape in shapes[1:]):
                dynamic_dims[name].append(idx)

    if not any(any(dict_.values()) for dict_ in [dynamic_dims, reshaped_static_dims]):
        return model

    partial_shapes = {}
    for name, shape in model_inputs_shapes.items():
        dataset_first_shape = dataset_inputs_shapes[name].pop()
        dims = []
        for idx, d in enumerate(shape):
            if idx in dynamic_dims[name]:
                dim = Dimension(-1)
            elif idx in reshaped_static_dims[name]:
                dim = Dimension(dataset_first_shape[idx])
            else:
                if isinstance(d, Dimension):
                    dim = d
                elif isinstance(d, tuple):
                    dim = Dimension(d[0], d[1])
                else:
                    dim = Dimension(d)
            dims.append(dim)
        partial_shapes[name] = PartialShape(dims)
    model.reshape(partial_shapes)
    return model


# pylint: disable=protected-access
def quantize_model(xml_path, bin_path, accuracy_checker_config, quantization_impl, quantization_parameters):
    ov_model = ov.Core().read_model(model=xml_path, weights=bin_path)
    model_evaluator = create_model_evaluator(accuracy_checker_config)
    model_evaluator.load_network([{"model": ov_model}])
    model_evaluator.select_dataset("")

    advanced_parameters = quantization_parameters.get("advanced_parameters", AdvancedQuantizationParameters())
    if quantization_impl == "pot":
        advanced_parameters.backend_params["use_pot"] = True
    elif quantization_impl == "native":
        advanced_parameters.backend_params["use_pot"] = False
    else:
        raise NotImplementedError()
    quantization_parameters["advanced_parameters"] = advanced_parameters

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(batch_input, batch_annotation)
        input_data = {}
        for name, value in filled_inputs[0].items():
            input_data[model_evaluator.launcher.input_to_tensor_name[name]] = value
        return input_data

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)

    if get_allow_reshape_input(accuracy_checker_config):
        ov_model = maybe_reshape_model(
            ov_model,
            calibration_dataset,
            quantization_parameters.get("subset_size", 300),
            model_evaluator.launcher.input_to_tensor_name,
        )
        model_evaluator.load_network([{"model": ov_model}])

    quantized_model = nncf.quantize(ov_model, calibration_dataset, **quantization_parameters)
    return quantized_model


# pylint: disable=protected-access
def quantize_model_with_accuracy_control(
    xml_path: str, bin_path: str, accuracy_checker_config, quantization_impl: str, quantization_parameters
):
    ov_model = ov.Core().read_model(xml_path, bin_path)
    model_evaluator = create_model_evaluator(accuracy_checker_config)
    model_evaluator.load_network_from_ir([{"model": xml_path, "weights": bin_path}])
    model_evaluator.select_dataset("")

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(batch_input, batch_annotation)
        return filled_inputs[0]

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    validation_dataset = nncf.Dataset(list(range(model_evaluator.dataset.full_size)))

    if get_allow_reshape_input(accuracy_checker_config):
        ov_model = maybe_reshape_model(
            ov_model,
            calibration_dataset,
            quantization_parameters.get("subset_size", 300),
            model_evaluator.launcher.input_to_tensor_name,
        )
        model_evaluator.load_network([{"model": ov_model}])

    metric_name = accuracy_checker_config["models"][0]["datasets"][0]["metrics"][0].get("name", None)
    if metric_name is None:
        metric_name = accuracy_checker_config["models"][0]["datasets"][0]["metrics"][0]["type"]
    validation_fn = ACValidationFunction(model_evaluator, metric_name)

    name_to_quantization_impl_map = {
        "pot": pot_quantize_with_native_accuracy_control,
        "native": nncf.quantize_with_accuracy_control,
    }

    advanced_parameters = quantization_parameters.get(
        "advanced_quantization_parameters", AdvancedQuantizationParameters()
    )
    if quantization_impl == "native":
        advanced_parameters.backend_params["use_pot"] = False
    quantization_parameters["advanced_quantization_parameters"] = advanced_parameters

    quantization_impl_fn = name_to_quantization_impl_map.get(quantization_impl)
    if quantization_impl:
        quantized_model = quantization_impl_fn(
            ov_model, calibration_dataset, validation_dataset, validation_fn, **quantization_parameters
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {quantization_impl}")

    return quantized_model


def main():
    args = parse_args()
    config = Config.read_config(args.config)
    config.configure_params()

    xml_path, bin_path = get_model_paths(config.model)
    accuracy_checker_config = get_accuracy_checker_config(config.engine)
    nncf_algorithms_config = get_nncf_algorithms_config(config.compression)

    set_log_file(f"{args.output_dir}/log.txt")
    output_dir = os.path.join(args.output_dir, "optimized")
    os.makedirs(output_dir, exist_ok=True)

    algo_name_to_method_map = {
        "quantize": quantize_model,
        "quantize_with_accuracy_control": quantize_model_with_accuracy_control,
    }
    for algo_config in nncf_algorithms_config:
        algo_name = algo_config["name"]
        algo_fn = algo_name_to_method_map.get(algo_name, None)
        if algo_fn:
            quantize_model_arguments = {
                "xml_path": xml_path,
                "bin_path": bin_path,
                "accuracy_checker_config": accuracy_checker_config,
                "quantization_impl": args.impl,
                "quantization_parameters": algo_config["parameters"],
            }

            output_model = algo_fn(**quantize_model_arguments)

            path = os.path.join(output_dir, "algorithm_parameters.json")
            keys = ["xml_path", "quantization_impl", "quantization_parameters"]
            dump_to_json(path, quantize_model_arguments, keys)
        else:
            raise RuntimeError(f"Support for {algo_name} is not implemented in the optimize tool.")

    model_name = config.model.model_name
    output_model_path = os.path.join(output_dir, f"{model_name}.xml")
    ov.serialize(output_model, output_model_path)


if __name__ == "__main__":
    main()
