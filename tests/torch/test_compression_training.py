"""
 Copyright (c) 2019-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import re
import tempfile
from copy import deepcopy

import pytest
import torch
from pytest import approx

from examples.torch.common.utils import get_name
from nncf import NNCFConfig
from tests.common.helpers import TEST_ROOT
from tests.common.helpers import get_cli_dict_args
from tests.torch.helpers import Command
from tests.torch.test_sanity_sample import create_command_line
from tests.torch.test_sanity_sample import update_compression_algo_dict_with_legr_save_load_params

# pylint: disable=redefined-outer-name
# sample
# ├── dataset
# │   ├── path
# │   ├── batch
# │   ├── configs
# │   │     ├─── config_filename
# │   │     │       ├── expected_accuracy
# │   │     │       ├── absolute_tolerance_train
# │   │     │       ├── absolute_tolerance_eval
# │   │     │       ├── execution_arg
# │   │     │       ├── epochs
# │   │     │       ├── weights
GLOBAL_CONFIG = {
    'classification':
        {
            'cifar10':
                {
                    'configs': {
                        'mobilenet_v2_cifar10_nas_SMALL.json': {
                            'expected_accuracy': 81.59,
                            'train_steps': 2,
                            'weights': 'mobilenet_v2_cifar10_93.91.pth',
                            'absolute_tolerance_train': 1.0,
                            'absolute_tolerance_eval': 2e-2,
                            'is_experimental': True,
                            'main_filename': 'nas_advanced_main.py',
                            'checkpoint_name': 'supernet',
                        },
                        'resnet50_cifar10_nas_SMALL.json': {
                            'expected_accuracy': 72.76,
                            'train_steps': 2,
                            'weights': 'resnet50_cifar10_93.65.pth',
                            'absolute_tolerance_train': 2.0,
                            'absolute_tolerance_eval': 2e-2,
                            'is_experimental': True,
                            'main_filename': 'nas_advanced_main.py',
                            'checkpoint_name': 'supernet',
                        },
                        'vgg11_bn_cifar10_nas_SMALL.json': {
                            'expected_accuracy': 77.76,
                            'train_steps': 2,
                            'weights': 'vgg11_bn_cifar10_92.39.pth',
                            'absolute_tolerance_train': 2.0,
                            'absolute_tolerance_eval': 2e-2,
                            'is_experimental': True,
                            'main_filename': 'nas_advanced_main.py',
                            'checkpoint_name': 'supernet',
                        },
                    },
                },
            'cifar100':
                {
                    'configs': {
                        'mobilenet_v2_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 68.11,
                            'weights': 'mobilenet_v2_32x32_cifar100_68.11.pth',
                            'absolute_tolerance_train': 1.0,
                            'absolute_tolerance_eval': 2e-2
                        },
                        'mobilenet_v2_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed', 'cpu-only'},
                            'expected_accuracy': 68.11,
                            'weights': 'mobilenet_v2_32x32_cifar100_68.11.pth',
                            'absolute_tolerance_train': 1.0,
                            'absolute_tolerance_eval': 6e-2
                        },
                        'inceptionV3_int8.json': {
                            'expected_accuracy': 77.53,
                            'weights': 'inceptionV3_77.53.sd',
                            'absolute_tolerance_eval': 6e-2,
                        },
                        'resnet50_int8.json': {
                            'expected_accuracy': 67.93,
                            'weights': 'resnet50_cifar100_67.93.pth',
                            'absolute_tolerance_eval': 6e-2,
                        },
                        'mobilenet_v2_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 68.11,
                            'weights': 'mobilenet_v2_32x32_cifar100_68.11.pth',
                            'execution_arg': {'multiprocessing-distributed', ''},
                            'absolute_tolerance_train': 1.5,
                            'absolute_tolerance_eval': 6e-2
                        },
                        'mobilenet_v2_rb_sparsity_int8.json': {
                            'expected_accuracy': 68.11,
                            'weights': 'mobilenet_v2_32x32_cifar100_68.11.pth',
                            'execution_arg': {'multiprocessing-distributed'},
                            'absolute_tolerance_eval': 2e-2
                        },
                        'mobilenet_v2_learned_ranking.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 68.11,
                            'weights': 'mobilenet_v2_32x32_cifar100_68.11.pth',
                            'absolute_tolerance_train': 1.5,
                            'absolute_tolerance_eval': 2e-2
                        },
                        'efficient_net_b0_cifar10_nas_SMALL.json': {
                            # TODO: fix supernet training (ticket 73814)
                            'expected_accuracy': 10.04,
                            'train_steps': 2,
                            'weights': 'efficient_net_b0_cifar100_87.02.pth',
                            'absolute_tolerance_train': 1.0,
                            'absolute_tolerance_eval': 2e-2,
                            'is_experimental': True,
                            'main_filename': 'nas_advanced_main.py',
                            'checkpoint_name': 'supernet',
                        },
                    }
                },
            'imagenet':
                {
                    'configs': {
                        'mobilenet_v2_imagenet_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                            'weights': 'mobilenet_v2.pth.tar'
                        },
                        'mobilenet_v2_imagenet_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                            'weights': 'mobilenet_v2.pth.tar',
                        },
                        'resnet50_imagenet_sym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                        },
                        'resnet50_imagenet_asym_int8.json': {
                            'execution_arg': {'multiprocessing-distributed'},
                            'expected_accuracy': 100,
                        },
                    }
                }
        }
}


def update_compression_config_with_legr_save_load_params(nncf_config_path, tmp_path, save=True):
    nncf_config = NNCFConfig.from_json(nncf_config_path)
    updated_nncf_config = update_compression_algo_dict_with_legr_save_load_params(deepcopy(nncf_config), tmp_path, save)
    new_nncf_config_path = nncf_config_path
    if updated_nncf_config != nncf_config:
        new_nncf_config_path = os.path.join(tmp_path, os.path.basename(nncf_config_path))
        with open(str(new_nncf_config_path), 'w', encoding='utf8') as f:
            json.dump(updated_nncf_config, f)
    return new_nncf_config_path


def parse_best_acc1(tmp_path):
    output_path = None
    for root, _, names in os.walk(str(tmp_path)):
        for name in names:
            if 'output' in name:
                output_path = os.path.join(root, name)

    assert os.path.exists(output_path)
    with open(output_path, "r", encoding='utf8') as f:
        for line in reversed(f.readlines()):
            if line != '\n':
                matches = re.findall("\\d+\\.\\d+", line)
                if not matches:
                    raise RuntimeError("Could not parse output log for accuracy!")
                acc1 = float(matches[0])
                return acc1
    raise RuntimeError("Could not parse output log for accuracy!")


CONFIG_PARAMS = []
for sample_type_, datasets in GLOBAL_CONFIG.items():
    for dataset_name_, dataset in datasets.items():
        dataset_path = dataset.get('path', os.path.join(tempfile.gettempdir(), dataset_name_))
        batch_size = dataset.get('batch', None)
        configs = dataset.get('configs', {})
        for config_name in configs:
            config_params = configs[config_name]
            main_filename = config_params.get('main_filename', 'main.py')
            is_experimental = config_params.get('is_experimental', False)
            test_timeout = config_params.get('test_timeout')
            train_steps = config_params.get('train_steps', None)
            execution_args = config_params.get('execution_arg', [''])
            expected_accuracy_ = config_params.get('expected_accuracy', 100)
            absolute_tolerance_train_ = config_params.get('absolute_tolerance_train', 1)
            absolute_tolerance_eval_ = config_params.get('absolute_tolerance_eval', 1e-3)
            weights_path_ = config_params.get('weights', None)
            epochs = config_params.get('epochs', None)
            if weights_path_:
                weights_path_ = os.path.join(sample_type_, dataset_name_, weights_path_)
            for execution_arg_ in execution_args:
                config_path_ = TEST_ROOT.joinpath(
                    "torch", "data", "configs", "weekly", sample_type_, dataset_name_, config_name)
                jconfig = NNCFConfig.from_json(config_path_)
                args_ = {
                    'data': dataset_path,
                    'weights': weights_path_,
                    'config': str(config_path_)
                }
                checkpoint_name = config_params.get('checkpoint_name', get_name(jconfig))
                if batch_size:
                    args_['batch-size'] = batch_size
                if epochs:
                    args_['epochs'] = epochs
                if train_steps:
                    args_['train-steps'] = train_steps
                test_config_ = {
                    'sample_type': sample_type_,
                    'main_filename': main_filename,
                    'is_experimental': is_experimental,
                    'timeout': test_timeout,
                    'expected_accuracy': expected_accuracy_,
                    'absolute_tolerance_train': absolute_tolerance_train_,
                    'absolute_tolerance_eval': absolute_tolerance_eval_,
                    'checkpoint_name': checkpoint_name
                }
                CONFIG_PARAMS.append(tuple([test_config_, args_, execution_arg_, dataset_name_]))


def get_config_name(config_path):
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([p[0]['sample_type'], get_config_name(p[1]['config']), p[2]]) for p in CONFIG_PARAMS])
def _params(request, tmp_path_factory, dataset_dir, weekly_models_path, enable_imagenet):
    if weekly_models_path is None:
        pytest.skip('Path to models weights for weekly testing is not set, use --weekly-models option.')
    test_config, args, execution_arg, dataset_name = request.param
    if test_config.get('timeout') is None:
        test_config['timeout'] = 3 * 60 * 60  # 3 hours, because legr training takes 2.5-3 hours
    if enable_imagenet:
        test_config['timeout'] = None
    if 'imagenet' in dataset_name and not enable_imagenet:
        pytest.skip('ImageNet tests were intentionally skipped as it takes a lot of time')
    if args['weights']:
        weights_path = os.path.join(weekly_models_path, args['weights'])
        if not os.path.exists(weights_path):
            raise FileExistsError('Weights file does not exist: {}'.format(weights_path))
        args['weights'] = weights_path
    else:
        del args['weights']
    if execution_arg:
        args[execution_arg] = None
    checkpoint_save_dir = str(tmp_path_factory.mktemp('models'))
    checkpoint_save_dir = os.path.join(checkpoint_save_dir, execution_arg.replace('-', '_'))
    args['checkpoint-save-dir'] = checkpoint_save_dir
    if dataset_dir:
        args['data'] = dataset_dir
    return {
        'test_config': test_config,
        'args': args,
    }


@pytest.fixture(scope="module")
def case_common_dirs(tmp_path_factory):
    return {
        "save_coeffs_path": str(tmp_path_factory.mktemp("ranking_coeffs")),
    }


@pytest.mark.dependency(name="train")
def test_compression_train(_params, tmp_path, case_common_dirs):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['config'] = update_compression_config_with_legr_save_load_params(args['config'],
                                                                          case_common_dirs["save_coeffs_path"])

    args['mode'] = 'train'
    args['log-dir'] = tmp_path
    args['workers'] = 0  # Workaround for PyTorch MultiprocessingDataLoader issues
    args['seed'] = 1
    # Workaround for PyTorch 1.9.1 Multiprocessing issue related to determinism and asym quantization
    # https://github.com/pytorch/pytorch/issues/61032
    if 'mobilenet_v2_asym_int8.json' in args['config']:
        args.pop('seed')

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type'], tc['main_filename'],
                                         tc['is_experimental']))
    env_with_cuda_reproducibility = os.environ.copy()
    env_with_cuda_reproducibility['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    runner.kwargs.update(env=env_with_cuda_reproducibility)
    runner.run(timeout=tc['timeout'])

    checkpoint_path = os.path.join(args['checkpoint-save-dir'], tc['checkpoint_name'] + '_best.pth')
    assert os.path.exists(checkpoint_path)
    actual_acc = torch.load(checkpoint_path)['best_acc1']
    ref_acc = tc['expected_accuracy']
    better_accuracy_tolerance = 3
    tolerance = tc['absolute_tolerance_train'] if actual_acc < ref_acc else better_accuracy_tolerance
    assert actual_acc == approx(ref_acc, abs=tolerance)


@pytest.mark.dependency(depends=["train"])
def test_compression_eval_trained(_params, tmp_path, case_common_dirs):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['config'] = update_compression_config_with_legr_save_load_params(args['config'],
                                                                          case_common_dirs["save_coeffs_path"], False)
    args['mode'] = 'test'
    args['log-dir'] = tmp_path
    args['workers'] = 0  # Workaround for PyTorch MultiprocessingDataLoader issues
    args['seed'] = 1
    # Workaround for PyTorch 1.9.1 Multiprocessing issue related to determinism and asym quantization
    # https://github.com/pytorch/pytorch/issues/61032
    if 'mobilenet_v2_asym_int8.json' in args['config']:
        args.pop('seed')
    checkpoint_path = os.path.join(args['checkpoint-save-dir'], tc['checkpoint_name'] + '_best.pth')
    args['resume'] = checkpoint_path

    METRIC_FILE_PATH = tmp_path / 'metrics.json'
    args['metrics-dump'] = tmp_path / METRIC_FILE_PATH

    if 'weights' in args:
        del args['weights']

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type'], tc['main_filename'],
                                         tc['is_experimental']))
    env_with_cuda_reproducibility = os.environ.copy()
    env_with_cuda_reproducibility['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    runner.kwargs.update(env=env_with_cuda_reproducibility)
    runner.run(timeout=tc['timeout'])

    with open(str(METRIC_FILE_PATH), encoding='utf8') as metric_file:
        metrics = json.load(metric_file)
    acc1 = metrics['Accuracy']
    assert torch.load(checkpoint_path)['best_acc1'] == approx(acc1, abs=tc['absolute_tolerance_eval'])
