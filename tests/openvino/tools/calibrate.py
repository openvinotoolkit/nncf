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

import json
import os
from argparse import ArgumentParser
from typing import Optional, TypeVar

import openvino.runtime as ov
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import \
    create_model_evaluator
from openvino.tools.pot.configs.config import Config

import nncf
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data.dataset import Dataset
from nncf.experimental.openvino_native.quantization.quantize import \
    quantize_impl
from nncf.scopes import IgnoredScope
from nncf.scopes import convert_ignored_scope_to_list
from nncf.parameters import (
    ModelType,
    TargetDevice
)

TModel = TypeVar('TModel')

MAP_POT_NNCF_ALGORITHMS = {'DefaultQuantization': 'quantize'}


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
            return ','.join(convert_ignored_scope_to_list(o))
        raise TypeError(f'Object of type {o.__class__.__name__} '
                        f'is not JSON serializable')


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
    if operations is not None:
        for op in operations:
            if op.get('attributes') is not None:
                raise ValueError('Attributes in the ignored operations '
                                 'are not supported')
    return {'ignored_scope': nncf.IgnoredScope(names=ignored.get('scope'),
                                               types=ignored.get('types'))}


def map_preset(preset):
    preset = preset.lower()
    if preset not in [p.value for p in nncf.QuantizationPreset]:
        raise ValueError(f'{preset} preset is not supported')
    return {'preset': nncf.QuantizationPreset(preset)}


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


def map_paramaters(pot_algo_name, nncf_algo_name, pot_parameters):
    if pot_algo_name == 'DefaultQuantization' and nncf_algo_name == 'quantize':
        return map_quantization_parameters(pot_parameters)
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


def main():
    args = parse_args()
    config = Config.read_config(args.config)
    config.configure_params()

    xml_path, bin_path = get_model_paths(config.model)
    accuracy_checcker_config = get_accuracy_checker_config(config.engine)
    nncf_algorithms_config = get_nncf_algorithms_config(config.compression)

    output_dir = os.path.join(args.output_dir, 'optimized')
    os.makedirs(output_dir, exist_ok=True)

    for algo_config in nncf_algorithms_config:
        algo_name = algo_config['name']
        if algo_name == 'quantize':
            quantize_model_arguments = {
                'xml_path': xml_path,
                'bin_path': bin_path,
                'accuracy_checcker_config': accuracy_checcker_config,
                'quantization_impl': args.impl,
                    'quantization_parameters': algo_config['parameters']
            }

            output_model = quantize_model(**quantize_model_arguments)

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
