"""
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
"""

from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import asdict
import json
import multiprocessing
import os
from typing import Iterable, Optional, TypeVar

import numpy as np
import openvino.runtime as ov
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator
from openvino.tools.pot.configs.config import Config

import nncf
from nncf.common.logging.logger import set_log_file
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data.dataset import Dataset
from nncf.experimental.openvino.quantization.quantize import \
    quantize_with_accuracy_control as pot_quantize_with_native_accuracy_control
from nncf.experimental.openvino_native.quantization.quantize import quantize_impl
from nncf.experimental.openvino_native.quantization.quantize import \
    quantize_with_accuracy_control as native_quantize_with_native_accuracy_control
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.scopes import IgnoredScope

TModel = TypeVar('TModel')


MAP_POT_NNCF_ALGORITHMS = {
    'DefaultQuantization': 'quantize',
    'AccuracyAwareQuantization': 'quantize_with_accuracy_control',
}


def parse_args():
    """
    Parses command line arguments.

    :return: A dict with command-line arguments
    """
    parser = ArgumentParser(description='NNCF OpenVINO Benchmarking Tool',
                            allow_abbrev=False)

    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with optimization parameters (POT format).')

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='The directory where models are saved. Default: ./results')

    parser.add_argument(
        '--impl',
        help='NNCF OpenVINO backend implementation.',
        choices=['pot', 'native'],
        default=None
    )

    return parser.parse_args()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (nncf.TargetDevice, nncf.ModelType,
                          nncf.QuantizationPreset)):
            return o.value
        if isinstance(o, (nncf.IgnoredScope)):
            return asdict(o)
        raise TypeError(f'Object of type {o.__class__.__name__} '
                        f'is not JSON serializable')


class ACValidationFunction:
    """
    Implementation of a validation function using the Accuracy Checker.
    """

    def __init__(self,
                 model_evaluator: ModelEvaluator,
                 metric_name: str,
                 requests_number: Optional[int] = None):
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
            'subset': indices,
            'check_progress': False,
            'dataset_tag': '',
            'calculate_metrics': True,
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
            if metric.meta.get('target', 'higher-better') == 'higher-worse':
                sign = -1.0

            if metric.meta.get('calculate_mean', True):
                metric_value = np.mean(metric.evaluated_value)
            else:
                metric_value = metric.evaluated_value[0]

            metrics[metric.name] = sign * metric_value

        self._model_evaluator.reset()

        return metrics[self._metric_name]

    @staticmethod
    def _set_requests_number(params, requests_number):
        if requests_number:
            params['nreq'] = np.clip(requests_number, 1, multiprocessing.cpu_count())
            if params['nreq'] != requests_number:
                print('Number of requests {} is out of range [1, {}]. Will be used {}.'
                      .format(requests_number, multiprocessing.cpu_count(), params['nreq']))


def dump_to_json(path, value, keys):
    dump_value = {}
    for k in keys:
        dump_value[k] = value[k]

    with open(path, 'w', encoding='utf8') as f:
        json.dump(dump_value, f, cls=CustomJSONEncoder)


def map_target_device(target_device):
    target_device = target_device.upper()
    if target_device not in [t.value for t in nncf.TargetDevice]:
        raise ValueError(f'{target_device} target device is not supported')
    return {'target_device': nncf.TargetDevice(target_device)}


def map_model_type(model_type):
    if model_type is None:
        return None

    model_type = model_type.lower()
    if model_type not in [m.value for m in nncf.ModelType]:
        raise ValueError(f'{model_type} model type is not supported')
    return {'model_type': nncf.ModelType(model_type)}


def map_ignored_scope(ignored):
    if ignored is None:
        return None

    if ignored.get('skip_model') is not None:
        raise ValueError('skip_model attribute in the ignored tag is not '
                         'supported')

    operations  = ignored.get('operations')
    ignored_operations = []
    if operations is not None:
        for op in operations:
            if op.get('attributes') is not None:
                raise ValueError('Attributes in the ignored operations '
                                 'are not supported')
            ignored_operations.append(op['type'])
    return {'ignored_scope': nncf.IgnoredScope(names=ignored.get('scope'),
                                               types=ignored_operations)}


def map_preset(preset):
    preset = preset.lower()
    if preset not in [p.value for p in nncf.QuantizationPreset]:
        raise ValueError(f'{preset} preset is not supported')
    return {'preset': nncf.QuantizationPreset(preset)}


def create_parameters_for_algorithm(pot_parameters, supported_parameters, default_parameters, ignored_parameters):
    result = {}
    for name in pot_parameters:
        if (name in ignored_parameters or
            (name in default_parameters and
             pot_parameters[name] == default_parameters[name])):
            continue
        if name in supported_parameters:
            kwarg = supported_parameters[name](pot_parameters[name])
            if kwarg is not None:
                result.update(kwarg)
        else:
            raise ValueError(f'{name} parameter is not supported')

    return result


def map_quantization_parameters(pot_parameters):
    supported_parameters = {
        'target_device': map_target_device,
        'model_type': map_model_type,
        'ignored': map_ignored_scope,
        'preset': map_preset,
        'stat_subset_size': lambda x: {'subset_size': x},
        'use_fast_bias': lambda x: {'fast_bias_correction': x}
    }

    default_parameters = {
      'use_layerwise_tuning': False
    }

    ignored_parameters = [
        'dump_intermediate_model',
        'inplace_statistics',
        'num_samples_for_tuning',
        'batch_size',
        'optimizer',
        'loss',
        'tuning_iterations',
        'random_seed',
        'use_ranking_subset',
        'calibration_indices_pool',
        'calculate_grads_on_loss_increase_only',
        'weight_decay'
    ]

    result = create_parameters_for_algorithm(pot_parameters, supported_parameters,
                                             default_parameters, ignored_parameters)

    return result


def map_quantize_with_accuracy_control_parameters(pot_parameters):
    supported_parameters = {
        'target_device': map_target_device,
        'model_type': map_model_type,
        'ignored': map_ignored_scope,
        'preset': map_preset,
        'stat_subset_size': lambda x: {'subset_size': x},
        'use_fast_bias': lambda x: {'fast_bias_correction': x},
        # Accuracy control parameters
        'maximal_drop': lambda x: {'max_drop': x},
        'max_iter_num': lambda x: {'max_num_iterations': x},
    }

    default_parameters = {}

    ignored_parameters = [
        'dump_intermediate_model',
        'inplace_statistics',
        'activations',
        'weights',
        # Accuracy control parameters
        'ranking_subset_size',
        'drop_type',
        'use_prev_if_drop_increase',
        'base_algorithm',
        'annotation_free',
        'annotation_conf_threshold',
        'convert_to_mixed_preset',
        'metrics',
        'metric_subset_ratio',
        'tune_hyperparams',
    ]

    result = create_parameters_for_algorithm(pot_parameters, supported_parameters,
                                             default_parameters, ignored_parameters)

    return result


def map_paramaters(pot_algo_name, nncf_algo_name, pot_parameters):
    if pot_algo_name == 'DefaultQuantization' and nncf_algo_name == 'quantize':
        return map_quantization_parameters(pot_parameters)
    if pot_algo_name == 'AccuracyAwareQuantization' and nncf_algo_name == 'quantize_with_accuracy_control':
        return map_quantize_with_accuracy_control_parameters(pot_parameters)
    raise ValueError(f'Mapping POT {pot_algo_name} parameters to NNCF '
                     f'{nncf_algo_name} parameters is not supported')


def get_model_paths(model_config):
    if model_config.cascade:
        raise ValueError('Cascade models are not supported yet.')
    return model_config.model, model_config.weights


def get_accuracy_checker_config(engine_config):
    if engine_config.type != 'accuracy_checker':
        raise ValueError(f'Engine type {engine_config.type} is not supported.')
    return engine_config


def get_nncf_algorithms_config(compression_config):
    nncf_algorithms = []
    for pot_algo in compression_config.algorithms:
        if pot_algo.name not in MAP_POT_NNCF_ALGORITHMS:
            raise ValueError(f'Algorithm {pot_algo.name} is not supported.')

        nncf_algo_name = MAP_POT_NNCF_ALGORITHMS[pot_algo.name]
        nncf_algorithms.append(
            {
                'name': nncf_algo_name,
                'parameters': map_paramaters(pot_algo.name, nncf_algo_name,
                                             pot_algo.params)
            }
        )
    return nncf_algorithms


def quantize_native(model: TModel,
                    calibration_dataset: Dataset,
                    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                    target_device: TargetDevice = TargetDevice.ANY,
                    subset_size: int = 300,
                    fast_bias_correction: bool = True,
                    model_type: Optional[ModelType] = None,
                    ignored_scope: Optional[IgnoredScope] = None) -> TModel:
    return quantize_impl(model, calibration_dataset, preset, target_device,
                         subset_size, fast_bias_correction, model_type,
                         ignored_scope)


# pylint: disable=protected-access
def quantize_model(xml_path, bin_path, accuracy_checcker_config,
                   quantization_impl, quantization_parameters):
    ov_model = ov.Core().read_model(model=xml_path, weights=bin_path)
    model_evaluator = create_model_evaluator(accuracy_checcker_config)
    model_evaluator.load_network([{'model': ov_model}])
    model_evaluator.select_dataset('')

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(
            batch_input, batch_annotation)
        return filled_inputs[0]

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    if quantization_impl == 'pot':
        quantized_model = nncf.quantize(ov_model, calibration_dataset,
                                        **quantization_parameters)
    elif quantization_impl == 'native':
        quantized_model = quantize_native(ov_model, calibration_dataset,
                                          **quantization_parameters)
    else:
        raise NotImplementedError()
    return quantized_model


# pylint: disable=protected-access
def quantize_model_with_accuracy_control(xml_path: str,
                                         bin_path: str,
                                         accuracy_checcker_config,
                                         quantization_impl: str,
                                         quantization_parameters):
    ov_model = ov.Core().read_model(xml_path, bin_path)
    model_evaluator = create_model_evaluator(accuracy_checcker_config)
    model_evaluator.load_network_from_ir([{'model': xml_path, 'weights': bin_path}])
    model_evaluator.select_dataset('')

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(
            batch_input, batch_annotation)
        return filled_inputs[0]

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    validation_dataset = nncf.Dataset(list(range(model_evaluator.dataset.full_size)))

    metric_name = accuracy_checcker_config['models'][0]['datasets'][0]['metrics'][0].get('name', None)
    if metric_name is None:
        metric_name = accuracy_checcker_config['models'][0]['datasets'][0]['metrics'][0]['type']
    validation_fn = ACValidationFunction(model_evaluator, metric_name)

    name_to_quantization_impl_map = {
        'pot': pot_quantize_with_native_accuracy_control,
        'native': native_quantize_with_native_accuracy_control,
    }

    quantization_impl_fn = name_to_quantization_impl_map.get(quantization_impl)
    if quantization_impl:
        quantized_model = quantization_impl_fn(ov_model, calibration_dataset, validation_dataset,
                                               validation_fn, **quantization_parameters)
    else:
        raise NotImplementedError(f'Unsupported implementation: {quantization_impl}')

    return quantized_model


def main():
    args = parse_args()
    config = Config.read_config(args.config)
    config.configure_params()

    xml_path, bin_path = get_model_paths(config.model)
    accuracy_checcker_config = get_accuracy_checker_config(config.engine)
    nncf_algorithms_config = get_nncf_algorithms_config(config.compression)

    set_log_file(f'{args.output_dir}/log.txt')
    output_dir = os.path.join(args.output_dir, 'optimized')
    os.makedirs(output_dir, exist_ok=True)

    algo_name_to_method_map = {
        'quantize': quantize_model,
        'quantize_with_accuracy_control': quantize_model_with_accuracy_control,
    }
    for algo_config in nncf_algorithms_config:
        algo_name = algo_config['name']
        algo_fn = algo_name_to_method_map.get(algo_name, None)
        if algo_fn:
            quantize_model_arguments = {
                'xml_path': xml_path,
                'bin_path': bin_path,
                'accuracy_checcker_config': accuracy_checcker_config,
                'quantization_impl': args.impl,
                    'quantization_parameters': algo_config['parameters']
            }

            output_model = algo_fn(**quantize_model_arguments)

            path = os.path.join(output_dir, 'algorithm_parameters.json')
            keys = ['xml_path',
                    'quantization_impl',
                    'quantization_parameters']
            dump_to_json(path, quantize_model_arguments, keys)
        else:
            raise RuntimeError(f'Support for {algo_name} is not implemented '
                               'in the optimize tool.')

    model_name = config.model.model_name
    output_model_path = os.path.join(output_dir, f'{model_name}.xml')
    ov.serialize(output_model, output_model_path)


if __name__ == '__main__':
    main()
