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


import functools
import json
import multiprocessing
import os
import platform
import subprocess
from argparse import ArgumentParser
from collections import OrderedDict
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, TypeVar

import numpy as np
import openvino.runtime as ov
import pkg_resources
from config import Config
from openvino.runtime import Dimension
from openvino.runtime import PartialShape

try:
    from accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator
    from accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
    from accuracy_checker.utils import extract_image_representations
except ImportError:
    from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator
    from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
    from openvino.tools.accuracy_checker.utils import extract_image_representations

import nncf
from nncf.common.deprecation import warning_deprecated
from nncf.common.logging import nncf_logger
from nncf.common.logging.logger import set_log_file
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.data.dataset import DataProvider
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AggregatorType
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import RestoreMode
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")

OVERRIDE_OPTIONS_ALGORITHMS = ["ActivationChannelAlignment", "FastBiasCorrection", "BiasCorrection"]

MAP_POT_NNCF_ALGORITHMS = {
    "ActivationChannelAlignment": {
        "method": "quantize",
        "advanced_parameters": {"disable_channel_alignment": False},
    },
    "FastBiasCorrection": {
        "method": "quantize",
        "advanced_parameters": {"disable_bias_correction": False},
        "parameters": {"fast_bias_correction": True},
    },
    "BiasCorrection": {
        "method": "quantize",
        "advanced_parameters": {"disable_bias_correction": False},
        "parameters": {"fast_bias_correction": False},
    },
    "MinMaxQuantization": {
        "method": "quantize",
        "advanced_parameters": {"disable_bias_correction": True, "disable_channel_alignment": True},
    },
    "DefaultQuantization": {"method": "quantize"},
    "AccuracyAwareQuantization": {"method": "quantize_with_accuracy_control"},
}

_default_context = None


def parse_args():
    """
    Parses command line arguments.

    :return: A dict with command-line arguments
    """
    parser = ArgumentParser(description="NNCF OpenVINO Benchmarking Tool", allow_abbrev=False)

    parser.add_argument("-c", "--config", help="Path to a config file with optimization parameters (JSON format).")

    parser.add_argument(
        "--output-dir", type=str, default="./results", help="The directory where models are saved. Default: ./results"
    )

    parser.add_argument("--impl", help="NNCF OpenVINO backend implementation.", choices=["pot", "native"], default=None)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=1)

    return parser.parse_args()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(
            o,
            (
                TargetDevice,
                ModelType,
                QuantizationPreset,
                OverflowFix,
                StatisticsType,
                AggregatorType,
                DropType,
                QuantizationMode,
                RestoreMode,
            ),
        ):
            return o.value
        if isinstance(o, (IgnoredScope, AdvancedQuantizationParameters, AdvancedAccuracyRestorerParameters)):
            return asdict(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class ACValidationFunction:
    """
    Implementation of a validation function using the Accuracy Checker.
    """

    METRIC_TO_PERSAMPLE_METRIC = {
        "coco_orig_precision": "coco_precision",
        "coco_orig_keypoints_precision": "coco_precision",
        "coco_orig_segm_precision": "coco_segm_precision",
        "hit_ratio": "sigmoid_recom_loss",
        "ndcg": "sigmoid_recom_loss",
    }

    SPECIAL_METRICS = [
        "cmc",
        "reid_map",
        "pairwise_accuracy_subsets",
        "pairwise_accuracy",
        "normalized_embedding_accuracy",
        "face_recognition_tafa_pair_metric",
        "localization_recall",
        "coco_orig_keypoints_precision",
        "coco_orig_segm_precision",
        "coco_orig_keypoints_precision",
        "spearman_correlation_coef",
        "pearson_correlation_coef",
    ]

    def __init__(
        self, model_evaluator: ModelEvaluator, metric_name: str, metric_type: str, requests_number: Optional[int] = None
    ):
        """
        :param model_evaluator: Model Evaluator.
        :param metric_name: Name of a metric.
        :param metric_type: Type of a metric.
        :param requests_number: A number of infer requests. If it is `None`,
            the count will be selected automatically.
        """
        self._model_evaluator = model_evaluator
        self._metric_name = metric_name
        self._metric_type = metric_type
        self._persample_metric_name = self.METRIC_TO_PERSAMPLE_METRIC.get(self._metric_name, self._metric_name)
        registered_metrics = model_evaluator.get_metrics_attributes()
        if self._persample_metric_name not in registered_metrics:
            self._model_evaluator.register_metric(self._persample_metric_name)
        self._requests_number = requests_number
        self._values_for_each_item = []

        self._collect_outputs = self._metric_type in self.SPECIAL_METRICS

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
            "output_callback": self._output_callback,
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

        values_for_each_item = sorted(self._values_for_each_item, key=lambda x: x["sample_id"])
        values_for_each_item = [x["metric_value"] for x in values_for_each_item]
        self._values_for_each_item = []

        return metrics[self._metric_name], values_for_each_item

    def _output_callback(self, raw_predictions, **kwargs):
        if not ("metrics_result" in kwargs and "dataset_indices" in kwargs):
            raise nncf.ValidationError(
                "Expected `metrics_result`, `dataset_indices` be passed to output_callback inside accuracy checker"
            )

        metrics_result = kwargs["metrics_result"]
        if metrics_result is None:
            return

        for sample_id, results in metrics_result.items():
            if self._collect_outputs:
                output = list(raw_predictions.values())[0]
                self._values_for_each_item.append({"sample_id": sample_id, "metric_value": output})
                continue

            for metric_result in results:
                if metric_result.metric_name != self._persample_metric_name:
                    continue

                sign = 1.0
                if metric_result.direction == "higher-worse":
                    sign = -1.0
                metric_value = sign * float(np.nanmean(metric_result.result))
                self._values_for_each_item.append({"sample_id": sample_id, "metric_value": metric_value})

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
    return {"ignored_scope": IgnoredScope(names=ignored.get("scope", []), types=ignored_operations)}


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
            quantization_params.mode = QuantizationScheme.SYMMETRIC
        elif mode == "asymmetric":
            quantization_params.mode = QuantizationScheme.ASYMMETRIC
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


def map_smooth_quant_alphas(smooth_quant_alphas):
    ctx = get_algorithm_parameters_context()
    advanced_parameter_name = ctx.param_name_map[ParameterNames.advanced_parameters]
    advanced_parameters = ctx.params.get(advanced_parameter_name, AdvancedQuantizationParameters())
    for key in ["convolution", "matmul"]:
        if key in smooth_quant_alphas:
            advanced_parameters.smooth_quant_alphas.__setattr__(key, smooth_quant_alphas[key])
    return {advanced_parameter_name: advanced_parameters}


def map_smooth_quant_alpha(smooth_quant_alpha):
    warning_deprecated(
        "`smooth_quant_alpha` parameter is deprecated."
        "Please, use `smooth_quant_alphas: {'convolution': .., 'matmul': ..}` instead."
    )
    return map_smooth_quant_alphas({"matmul": smooth_quant_alpha, "convolution": -1})


def map_mode(mode):
    if not hasattr(QuantizationMode, mode):
        raise ValueError(f"{mode} mode is not supported")
    return {"mode": getattr(QuantizationMode, mode)}


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


def map_dump_intermediate_model(dump_intermediate_model, output_dir):
    intermediate_model_dir = None

    if dump_intermediate_model:
        intermediate_model_dir = os.path.join(output_dir, "intermediate_model")
        os.makedirs(intermediate_model_dir, exist_ok=True)

    ctx = get_algorithm_parameters_context()
    advanced_parameters = ctx.params.get("advanced_accuracy_restorer_parameters", AdvancedAccuracyRestorerParameters())
    advanced_parameters.intermediate_model_dir = intermediate_model_dir
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
        "smooth_quant_alphas": map_smooth_quant_alphas,
        "smooth_quant_alpha": map_smooth_quant_alpha,
        "mode": map_mode,
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


def map_quantize_with_accuracy_control_parameters(pot_parameters, output_dir):
    supported_parameters, default_parameters, ignored_parameters = get_pot_quantization_parameters_mapping()

    ignored_parameters.remove("dump_intermediate_model")
    map_dump_intermediate_model_fn = functools.partial(map_dump_intermediate_model, output_dir=output_dir)

    supported_parameters.update(
        {
            "maximal_drop": lambda x: {"max_drop": x},
            "max_iter_num": map_max_iter_num,
            "ranking_subset_size": map_ranking_subset_size,
            "tune_hyperparams": map_tune_hyperparams,
            "drop_type": map_drop_type,
            "dump_intermediate_model": map_dump_intermediate_model_fn,
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
            "convert_to_mixed_preset",
        ]
    )

    param_name_map = {ParameterNames.advanced_parameters: "advanced_quantization_parameters"}

    result = create_parameters_for_algorithm(
        pot_parameters, supported_parameters, default_parameters, ignored_parameters, param_name_map
    )

    return result


def map_paramaters(pot_algo_name, nncf_algo_name, pot_parameters, output_dir):
    if nncf_algo_name == "quantize":
        return map_quantization_parameters(pot_parameters)
    if nncf_algo_name == "quantize_with_accuracy_control":
        return map_quantize_with_accuracy_control_parameters(pot_parameters, output_dir)
    raise ValueError(f"Mapping POT {pot_algo_name} parameters to NNCF {nncf_algo_name} parameters is not supported")


def get_model_paths(model_config):
    if model_config.cascade:
        raise ValueError("Cascade models are not supported yet.")
    return model_config.model, model_config.weights


def get_accuracy_checker_config(engine_config):
    if engine_config.type != "accuracy_checker":
        raise ValueError(f"Engine type {engine_config.type} is not supported.")
    return engine_config


def get_nncf_algorithms_config(compression_config, output_dir):
    nncf_algorithms = {}
    override_options = {}
    for pot_algo in compression_config.algorithms:
        pot_algo_name = pot_algo.name
        if pot_algo_name not in MAP_POT_NNCF_ALGORITHMS:
            raise ValueError(f"Algorithm {pot_algo_name} is not supported.")

        nncf_algo_name = MAP_POT_NNCF_ALGORITHMS[pot_algo_name]["method"]
        advanced_parameters = MAP_POT_NNCF_ALGORITHMS[pot_algo_name].get("advanced_parameters", None)
        parameters = MAP_POT_NNCF_ALGORITHMS[pot_algo_name].get("parameters", {})

        if pot_algo_name in OVERRIDE_OPTIONS_ALGORITHMS:
            if nncf_algo_name not in override_options:
                override_options[nncf_algo_name] = defaultdict(dict)

            override_options[nncf_algo_name]["advanced_parameters"].update(advanced_parameters)
            override_options[nncf_algo_name]["parameters"].update(parameters)
            continue

        nncf_algo_parameters = map_paramaters(pot_algo_name, nncf_algo_name, pot_algo.params, output_dir)

        if advanced_parameters is not None:
            nncf_algo_parameters["advanced_parameters"] = replace(
                nncf_algo_parameters["advanced_parameters"], **advanced_parameters
            )
        nncf_algorithms[nncf_algo_name] = nncf_algo_parameters

    for override_algo_name, override_values in override_options.items():
        nncf_algorithms[override_algo_name]["advanced_parameters"] = replace(
            nncf_algorithms[override_algo_name]["advanced_parameters"], **override_values["advanced_parameters"]
        )
        nncf_algorithms[override_algo_name].update(override_values["parameters"])
    return nncf_algorithms


def get_allow_reshape_input(accuracy_checker_config) -> bool:
    for model_config in accuracy_checker_config["models"]:
        for launcher_config in model_config["launchers"]:
            if "allow_reshape_input" in launcher_config:
                return launcher_config["allow_reshape_input"]
    return False


def maybe_reshape_model(model, dataset, subset_size, input_to_tensor_name):
    dataset_inputs_shapes = defaultdict(set)
    for input_dict in islice(dataset.get_inference_data(), subset_size):
        for name, tensor in input_dict.items():
            dataset_inputs_shapes[name].add(tuple(tensor.shape))

    model_inputs_shapes = {}
    for input_output in model.inputs:
        input_node = input_output.get_node()
        partial_shape = []
        for dim in input_node.partial_shape:
            partial_shape.append(Dimension(str(dim)))
        model_inputs_shapes[input_to_tensor_name[input_node.friendly_name]] = tuple(partial_shape)

    if len(dataset_inputs_shapes) != len(model_inputs_shapes):
        raise nncf.InternalError(
            f"Model inputs: {list(model_inputs_shapes.keys())}"
            f" and dataset inputs {list(dataset_inputs_shapes.keys())} are not compatible"
        )

    for name in model_inputs_shapes:
        if name not in dataset_inputs_shapes:
            raise nncf.ValidationError(
                f"Model input {name} is not present in dataset inputs: {list(dataset_inputs_shapes.keys())}"
            )

    dynamic_dims = defaultdict(list)
    reshaped_static_dims = defaultdict(list)
    for name, shapes in dataset_inputs_shapes.items():
        shapes = list(shapes)
        if len(set(len(shape) for shape in shapes)) != 1 or len(model_inputs_shapes[name]) != len(shapes[0]):
            raise nncf.InternalError("calibrate.py does not support dataset with dynamic ranks")

        for idx in range(len(shapes[0])):
            if len(shapes) == 1:
                model_dim = model_inputs_shapes[name][idx]
                if model_dim.is_static and model_dim.get_length() != shapes[0][idx]:
                    reshaped_static_dims[name].append(idx)

            elif any(shapes[0][idx] != shape[idx] for shape in shapes[1:]):
                dynamic_dims[name].append(idx)

    if not any(any(dict_.values()) for dict_ in [dynamic_dims, reshaped_static_dims]):
        return model, model_inputs_shapes

    partial_shapes = {}
    for name, partial_shape in model_inputs_shapes.items():
        dataset_first_shape = dataset_inputs_shapes[name].pop()
        dims = []
        for idx, d in enumerate(partial_shape):
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
    return model, model_inputs_shapes


def get_transform_fn(model_evaluator: ModelEvaluator, ov_model):
    if isinstance(model_evaluator, ModelEvaluator) and model_evaluator.launcher._lstm_inputs:
        compiled_original_model = ov.Core().compile_model(ov_model, device_name="CPU")
        model_outputs = None

        def transform_fn(data_item: ACDattasetWrapper.DataItem):
            model_inputs = data_item.data
            nonlocal model_outputs
            state_inputs = model_evaluator.launcher._fill_lstm_inputs(model_outputs)
            model_inputs.update(state_inputs)
            if data_item.status == ACDattasetWrapper.Status.SEQUENCE_IS_GOING:
                model_outputs = compiled_original_model(model_inputs)
            else:
                model_outputs = None
            return model_inputs

    else:

        def transform_fn(data_item: ACDattasetWrapper.DataItem):
            return data_item.data

    return transform_fn


def get_dataset(model_evaluator, quantization_parameters):
    dataset = ACDattasetWrapper(model_evaluator)
    sequence_subset_size = quantization_parameters.get("subset_size", 300)
    subset_size = dataset.calculate_per_sample_subset_size(sequence_subset_size)
    if subset_size != sequence_subset_size:
        print(f"Subset size is changed from {sequence_subset_size} to {subset_size}")
        print(f"Total dataset size: {len(dataset)}")
        quantization_parameters["subset_size"] = subset_size
    return dataset


class ACDattasetWrapper:
    """
    Iters through all items in sequences of model_evaluator dataset and
    returns DataItem with one of the status: sequence is going or end of sequence.
    """

    class Status(Enum):
        END_OF_SEQUENCE = "END_OF_SEQUENCE"
        SEQUENCE_IS_GOING = "SEQUENCE_IS_GOING"

    @dataclass
    class DataItem:
        data: Any
        status: "ACDattasetWrapper.Status"

    def __init__(self, model_evaluator):
        self.model_evaluator = model_evaluator
        self.batch_size = self.model_evaluator.dataset.batch

    def __iter__(self):
        if isinstance(self.model_evaluator, ModelEvaluator):
            for sequence in self.model_evaluator.dataset:
                _, batch_annotation, batch_input, _ = sequence
                filled_inputs, _, _ = self.model_evaluator._get_batch_input(batch_input, batch_annotation)
                for idx, filled_input in enumerate(filled_inputs):
                    input_data = {}
                    for name, value in filled_input.items():
                        input_data[self.model_evaluator.launcher.input_to_tensor_name[name]] = value
                    status = self.Status.SEQUENCE_IS_GOING
                    if idx == len(filled_inputs) - 1:
                        status = self.Status.END_OF_SEQUENCE
                    yield self.DataItem(input_data, status)
        else:
            for sequence in self.model_evaluator.dataset:
                _, batch_annotation, batch_input, _ = sequence
                batch_inputs = self.model_evaluator._internal_module.preprocessor.process(batch_input, batch_annotation)
                batch_data, _ = extract_image_representations(batch_inputs)
                input_data = np.expand_dims(batch_data[0], axis=0)
                yield self.DataItem(input_data, self.Status.END_OF_SEQUENCE)

    def __len__(self):
        return len(self.model_evaluator.dataset)

    def calculate_per_sample_subset_size(self, sequence_subset_size):
        if isinstance(self.model_evaluator, ModelEvaluator):
            subset_size = 0
            for data_item in islice(self.model_evaluator.dataset, sequence_subset_size):
                _, batch_annotation, batch_input, _ = data_item
                filled_inputs, _, _ = self.model_evaluator._get_batch_input(batch_input, batch_annotation)
                subset_size += len(filled_inputs)
            return subset_size
        else:
            return sequence_subset_size


def quantize_model(xml_path, bin_path, accuracy_checker_config, quantization_parameters):
    ov_model = ov.Core().read_model(model=xml_path, weights=bin_path)
    model_evaluator = create_model_evaluator(accuracy_checker_config)
    if isinstance(model_evaluator, ModelEvaluator):
        model_evaluator.load_network([{"model": ov_model}])
    model_evaluator.select_dataset("")

    advanced_parameters = quantization_parameters.get("advanced_parameters", AdvancedQuantizationParameters())
    if quantization_parameters.get("mode", None) is not None:
        advanced_parameters.backend_params = None
    quantization_parameters["advanced_parameters"] = advanced_parameters

    transform_fn = get_transform_fn(model_evaluator, ov_model)
    dataset = get_dataset(model_evaluator, quantization_parameters)
    calibration_dataset = nncf.Dataset(dataset, transform_fn)

    original_model_shapes = None
    if get_allow_reshape_input(accuracy_checker_config):
        ov_model, original_model_shapes = maybe_reshape_model(
            ov_model,
            calibration_dataset,
            quantization_parameters.get("subset_size", 300),
            model_evaluator.launcher.input_to_tensor_name,
        )
        model_evaluator.load_network([{"model": ov_model}])

    quantized_model = nncf.quantize(ov_model, calibration_dataset, **quantization_parameters)
    if original_model_shapes is not None:
        quantized_model.reshape(original_model_shapes)

    return quantized_model


class ACDataset:
    def __init__(self, model_evaluator, transform_func):
        self._model_evaluator = model_evaluator
        self._indices = list(range(model_evaluator.dataset.full_size))
        self._transform_func = transform_func

    def get_data(self, indices: Optional[List[int]] = None):
        return DataProvider(self._indices, None, indices)

    def get_inference_data(self, indices: Optional[List[int]] = None):
        return DataProvider(ACDattasetWrapper(self._model_evaluator), self._transform_func, indices)


def initialize_model_and_evaluator(xml_path: str, bin_path: str, accuracy_checker_config):
    model_evaluator = create_model_evaluator(accuracy_checker_config)

    model = ov.Core().read_model(xml_path, bin_path)
    model_evaluator.load_network_from_ir([{"model": xml_path, "weights": bin_path}])
    model_evaluator.select_dataset("")
    return model, model_evaluator


def quantize_model_with_accuracy_control(
    xml_path: str, bin_path: str, accuracy_checker_config, quantization_parameters
):
    ov_model, model_evaluator = initialize_model_and_evaluator(xml_path, bin_path, accuracy_checker_config)

    transform_fn = get_transform_fn(model_evaluator, ov_model)
    dataset = get_dataset(model_evaluator, quantization_parameters)
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    validation_dataset = ACDataset(model_evaluator, transform_fn)

    original_model_shapes = None
    if get_allow_reshape_input(accuracy_checker_config):
        ov_model, original_model_shapes = maybe_reshape_model(
            ov_model,
            calibration_dataset,
            quantization_parameters.get("subset_size", 300),
            model_evaluator.launcher.input_to_tensor_name,
        )
        model_evaluator.load_network([{"model": ov_model}])

    metric_type = accuracy_checker_config["models"][0]["datasets"][0]["metrics"][0]["type"]
    metric_name = accuracy_checker_config["models"][0]["datasets"][0]["metrics"][0].get("name", None)
    if metric_name is None:
        metric_name = metric_type
    validation_fn = ACValidationFunction(model_evaluator, metric_name, metric_type)

    advanced_parameters = quantization_parameters.get(
        "advanced_quantization_parameters", AdvancedQuantizationParameters()
    )
    quantization_parameters["advanced_quantization_parameters"] = advanced_parameters

    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model, calibration_dataset, validation_dataset, validation_fn, **quantization_parameters
    )

    if original_model_shapes is not None:
        quantized_model.reshape(original_model_shapes)
    return quantized_model


def filter_configuration(config: Config) -> Config:
    fields_to_filter = ["smooth_quant_alphas", "smooth_quant_alpha", "mode"]
    algorithms_to_update = defaultdict(dict)

    # Drop params before configure
    for algorithm_config in config["compression"]["algorithms"]:
        algo_params = algorithm_config.get("params")
        if algo_params is None:
            continue
        algo_name = algorithm_config.get("name")
        for field_to_filter in fields_to_filter:
            field_value = algo_params.get(field_to_filter)
            if field_value:
                del algo_params[field_to_filter]
                algorithms_to_update[algo_name][field_to_filter] = field_value

    config.configure_params()

    # Set dropped params
    for algorithm_config in config["compression"]["algorithms"]:
        algo_name = algorithm_config.get("name")
        if algo_name in algorithms_to_update:
            for field_name, field_value in algorithms_to_update[algo_name].items():
                algorithm_config["params"][field_name] = field_value

    return config


def update_accuracy_checker_config(accuracy_checker_config: Config, batch_size: int) -> None:
    """
    Updates batch section of accuracy checker configuration file by batch_size value.

    :param accuracy_checker_config: Accuracy checker configuration file.
    :param batch_size: Batch size value.
    """
    for model in accuracy_checker_config["models"]:
        for dataset in model["datasets"]:
            dataset["batch"] = batch_size
            print(f"Updated batch size value to {batch_size}")


def update_nncf_algorithms_config(nncf_algorithms_config: Dict[str, Dict[str, Any]], batch_size: int) -> None:
    """
    Updates subset_size parameter depending on batch_size and subset_size from an algorithm config.

    :param nncf_algorithms_config: Configuration file of an algorithm.
    :param batch_size: Batch size value.
    """
    for nncf_method, config in nncf_algorithms_config.items():
        subset_size = config.get("subset_size", 300)
        new_subset_size = subset_size // batch_size
        config["subset_size"] = new_subset_size
        print(f"Updated subset_size value for {nncf_method} method to {new_subset_size} ")


class EnvInfo:

    @staticmethod
    def print_info() -> None:
        """
        Prints NNCF version, python version and CPU model name.
        """
        python_version = EnvInfo._get_python_version()
        nncf_version = EnvInfo._get_nncf_version()
        cpu_model = EnvInfo._get_cpu_model()

        nncf_logger.info(f"Python version: {python_version}")
        nncf_logger.info(f"NNCF version: {nncf_version}")
        nncf_logger.info(f"CPU model: {cpu_model}")

    @staticmethod
    def _get_nncf_version() -> str:
        try:
            version = pkg_resources.get_distribution("nncf").version
        except pkg_resources.DistributionNotFound:
            version = "Unknown"
        return version

    @staticmethod
    def _get_python_version() -> str:
        return platform.python_version()

    @staticmethod
    def _get_cpu_model() -> str:
        """
        Returns CPU device name.
        """

        def _get_cpu_model_name_linux_os() -> str:
            try:
                output = subprocess.check_output("lscpu", shell=True)
                for line in output.decode().splitlines():
                    if "Model name:" in line:
                        return line.split(":")[1].strip()
                return "Unknown"
            except Exception:
                return "Unknown"

        def _get_cpu_model_name_windows() -> str:
            try:
                output = subprocess.check_output("wmic cpu get name", shell=True)
                return output.decode().splitlines()[1].strip()
            except Exception:
                return "Unknown"

        os_name = platform.system()

        cpu_model = "Unknown"
        if os_name == "Linux":
            cpu_model = _get_cpu_model_name_linux_os()
        elif os_name == "Windows":
            cpu_model = _get_cpu_model_name_windows()

        return cpu_model


def main():
    args = parse_args()
    if args.impl is not None:
        print("--impl option is deprecated and will have no effect. Only native calibration allowed.")
    config = Config.read_config(args.config)
    config = filter_configuration(config)

    xml_path, bin_path = get_model_paths(config.model)
    accuracy_checker_config = get_accuracy_checker_config(config.engine)
    nncf_algorithms_config = get_nncf_algorithms_config(config.compression, args.output_dir)
    assert args.batch_size >= 0
    if args.batch_size > 1:
        update_accuracy_checker_config(accuracy_checker_config, args.batch_size)
        update_nncf_algorithms_config(nncf_algorithms_config, args.batch_size)

    set_log_file(f"{args.output_dir}/log.txt")
    output_dir = os.path.join(args.output_dir, "optimized")
    os.makedirs(output_dir, exist_ok=True)

    EnvInfo.print_info()

    algo_name_to_method_map = {
        "quantize": quantize_model,
        "quantize_with_accuracy_control": quantize_model_with_accuracy_control,
    }
    for algo_name, algo_config in nncf_algorithms_config.items():
        algo_fn = algo_name_to_method_map.get(algo_name)
        if algo_fn:
            quantize_model_arguments = {
                "xml_path": xml_path,
                "bin_path": bin_path,
                "accuracy_checker_config": accuracy_checker_config,
                "quantization_parameters": algo_config,
            }

            output_model = algo_fn(**quantize_model_arguments)

            path = os.path.join(output_dir, "algorithm_parameters.json")
            keys = ["xml_path", "quantization_parameters"]
            dump_to_json(path, quantize_model_arguments, keys)
        else:
            raise nncf.InternalError(f"Support for {algo_name} is not implemented in the optimize tool.")

    model_name = config.model.model_name
    output_model_path = os.path.join(output_dir, f"{model_name}.xml")
    ov.serialize(output_model, output_model_path)


if __name__ == "__main__":
    main()
