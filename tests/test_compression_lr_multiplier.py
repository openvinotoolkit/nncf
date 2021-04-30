"""
 Copyright (c) 2019-2020 Intel Corporation
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

import contextlib
import copy
from typing import Iterable
from typing import Tuple
from typing import List
from typing import Dict

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD

from nncf import NNCFConfig
from nncf.layer_utils import CompressionParameter
from tests.helpers import create_initialized_compressed_model
from tests.helpers import create_random_mock_dataloader
from tests.helpers import check_equal
from tests.helpers import check_not_equal
from tests.helpers import check_less
from tests.helpers import check_greater
from tests.helpers import get_grads
from tests.helpers import LeNet
from tests.helpers import RandomDatasetMock
from tests.quantization.test_algo_quantization import get_quantization_config_without_range_init
from tests.sparsity.rb.test_algo import get_basic_sparsity_config


ALGO_NAME_TO_PATH_MAP = {
    'quantization': 'nncf.quantization',
    'rb_sparsity': 'nncf.sparsity.rb',
    'binarization': 'nncf.binarization'
}


@contextlib.contextmanager
def set_torch_seed(seed: int = 42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)


def get_quantization_config() -> NNCFConfig:
    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    config['compression']['initializer'] = {
        'range': {
            'num_init_samples': 10
        },
        'batchnorm_adaptation': {
            'num_bn_adaptation_samples': 0,
        }
    }
    return config


def get_sparsity_config() -> NNCFConfig:
    config = get_basic_sparsity_config([1, *LeNet.INPUT_SIZE])
    return config


def get_binarization_config() -> NNCFConfig:
    config = NNCFConfig()
    config.update({
        "model": "resnet18",

        "input_info": {
            "sample_size": [1, *LeNet.INPUT_SIZE]
        },

        "compression": [
            {
                "algorithm": "binarization",
                "mode": "xnor",
                "params": {
                    "activations_quant_start_epoch": 0,
                    "weights_quant_start_epoch": 0
                }
            }
        ]
    })
    return config


def get_config_algorithms(config: NNCFConfig) -> List[Dict]:
    if isinstance(config['compression'], list):
        algorithms = config['compression']
    else:
        algorithms = [config['compression']]
    return algorithms


def add_multiplier_to_config(config: NNCFConfig, local_multiplier: float = None, global_multiplier: float = None):
    config = copy.deepcopy(config)

    if local_multiplier is not None:
        algorithms = get_config_algorithms(config)

        for algo in algorithms:
            algo.update({
                'compression_lr_multiplier': local_multiplier
            })

    if global_multiplier is not None:
        config['compression_lr_multiplier'] = global_multiplier

    return config


def get_multipliers_from_config(config: NNCFConfig) -> Dict[str, float]:
    algo_to_multipliers = {}

    algorithms = get_config_algorithms(config)
    global_multiplier = config.get('compression_lr_multiplier', 1)
    for algo in algorithms:
        algo_name = algo['algorithm']
        algo_to_multipliers[algo_name] = algo.get('compression_lr_multiplier', global_multiplier)

    return algo_to_multipliers


def merge_configs(configs: List[NNCFConfig], use_algo_list: bool = True) -> NNCFConfig:
    res_config = None
    algorithms = []

    for source_config in configs:
        source_config = copy.deepcopy(source_config)

        algorithms.extend(get_config_algorithms(source_config))
        del source_config['compression']

        if res_config is None:
            res_config = source_config
        res_config.update(source_config)

    if not use_algo_list:
        if len(algorithms) > 1:
            raise Exception('If there is more than one algorithm '
                            'you could use only use_algo_list=True')
        res_config['compression'] = algorithms[0]
    else:
        res_config['compression'] = algorithms

    res_config['model'] = 'merged_model'
    return res_config


def get_configs_building_params() -> List[Dict]:
    res = []
    get_orig_config_fns = [get_quantization_config, get_sparsity_config, get_binarization_config]
    num_orig_configs = len(get_orig_config_fns)

    for global_multiplier in [0, 1, 10]:
        res.append({
                'get_orig_config_fns': get_orig_config_fns,
                'multipliers': [None] * num_orig_configs,
                'global_multiplier': global_multiplier,
                'use_algo_list': True
        })

    global_multiplier = 10
    multipliers = [global_multiplier * (1.1 ** i) for i in range(num_orig_configs)]

    res.append({
        'get_orig_config_fns': get_orig_config_fns,
        'multipliers': multipliers,
        'global_multiplier': global_multiplier,
        'use_algo_list': True
    })

    for i in range(num_orig_configs):
        cur_multipliers = copy.deepcopy(multipliers)
        cur_multipliers[i] = None
        res.append({
            'get_orig_config_fns': get_orig_config_fns,
            'multipliers': cur_multipliers,
            'global_multiplier': None,
            'use_algo_list': True
        })

    for get_orig_config_fn in get_orig_config_fns:
        for use_algo_list in [False, True]:
            for global_multiplier, multiplier in [(11, 10), (11, None), (None, 10)]:
                res.append({
                    'get_orig_config_fns': [get_orig_config_fn],
                    'multipliers': [multiplier],
                    'global_multiplier': global_multiplier,
                    'use_algo_list': use_algo_list
                })

    return res


@pytest.fixture(name='ref_and_target_configs',
                params=get_configs_building_params())
def ref_and_target_configs_(request) -> Tuple[NNCFConfig, NNCFConfig]:
    ref_configs = [get_ref_config_fn() for get_ref_config_fn in request.param['get_orig_config_fns']]
    ref_config = merge_configs(ref_configs, request.param['use_algo_list'])

    target_configs = [add_multiplier_to_config(config, local_multiplier=multiplier)
                      for config, multiplier in zip(ref_configs, request.param['multipliers'])]
    target_config = merge_configs(target_configs, request.param['use_algo_list'])
    target_config = add_multiplier_to_config(target_config, global_multiplier=request.param['global_multiplier'])

    return ref_config, target_config


class OneParameterModel(nn.Module):
    INPUT_SIZE = 0,

    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, _x):
        return self.param.sum()


def create_initialized_one_parameter_model(parameter_cls: type, init_requires_grad: bool,
                                           requires_grad_settings: List[Tuple[str, bool]],
                                           multiplier: float = None) -> nn.Module:
    with set_torch_seed():
        data = torch.randn(size=(1, 1, 5, 5))
        if parameter_cls is nn.Parameter:
            param = parameter_cls(data, requires_grad=init_requires_grad)
        elif parameter_cls is CompressionParameter:
            param = parameter_cls(data, requires_grad=init_requires_grad,
                                  compression_lr_multiplier=multiplier)
        else:
            raise Exception(f'Unsupported parameter type: {parameter_cls}')

    for setting_type, requires_grad in requires_grad_settings:
        if setting_type == 'attr':
            param.requires_grad = requires_grad
        elif setting_type == 'fn':
            param.requires_grad_(requires_grad)
        else:
            raise Exception(f'Unsupported setting type: {setting_type}')

    return OneParameterModel(param)


def get_one_parameter_model_creation_params(for_training: bool = False) -> List[Dict]:
    params = []
    for init_requires_grad in [False, True]:
        requires_grad_settings_list = [
            [], [('attr', False)], [('attr', True)], [('fn', False)], [('fn', True)],
            [('attr', not init_requires_grad), ('attr', True)], [('fn', not init_requires_grad), ('fn', True)],
            [('attr', not init_requires_grad), ('fn', True)], [('fn', not init_requires_grad), ('attr', True)]
        ]

        for requires_grad_settings in requires_grad_settings_list:
            trainable = init_requires_grad if len(requires_grad_settings) == 0 else requires_grad_settings[-1][1]
            if for_training and not trainable:
                continue
            multipliers = [0.1, 1, 10] if trainable else [0.1]

            for multiplier in multipliers:
                params.append({
                    'init_requires_grad': init_requires_grad,
                    'requires_grad_settings': requires_grad_settings,
                    'multiplier': multiplier
                })
    return params


def perform_model_training_steps(model: nn.Module, train_loader, num_steps: int = 1) -> nn.Module:
    with set_torch_seed():
        train_loader = iter(train_loader)
        optimizer = SGD(model.parameters(), lr=0.1)

        # This block of code is needed to initialize scale in the binarization algorithm
        # TODO: perform binarization scale init in the same way as for quantization
        with torch.no_grad():
            x, y_gt = next(train_loader)
            model(x)

        for _ in range(num_steps):
            optimizer.zero_grad()
            x, y_gt = next(train_loader)
            y = model(x)
            loss = F.mse_loss(y.sum(), y_gt)

            loss.backward()
            optimizer.step()

    return model


def get_params_grouped_by_algorithms(model: nn.Module) -> Dict[str, Iterable[nn.Parameter]]:
    cls_name_to_params = {}
    for module in model.modules():
        params = module.parameters(recurse=False)
        full_cls_name = '.'.join([module.__class__.__module__, module.__class__.__name__])
        if full_cls_name not in cls_name_to_params:
            cls_name_to_params[full_cls_name] = []
        cls_name_to_params[full_cls_name].extend(params)

    algo_name_to_params = {}
    for cls_name, params in cls_name_to_params.items():
        params = [param for param in params if param.requires_grad]
        if len(params) == 0:
            continue

        algo_name = 'regular'
        for cur_algo_name, cur_algo_path in ALGO_NAME_TO_PATH_MAP.items():
            if cur_algo_path in cls_name:
                algo_name = cur_algo_name

        if algo_name not in algo_name_to_params:
            algo_name_to_params[algo_name] = []
        algo_name_to_params[algo_name].extend(params)

    return algo_name_to_params


def get_lenet_params_after_training_steps(config: NNCFConfig, num_steps=1):
    with set_torch_seed():
        train_loader = create_random_mock_dataloader(config, num_samples=10)
        model = LeNet()
        for param in model.parameters():
            nn.init.normal_(param)

        model = create_initialized_compressed_model(model, config, train_loader)
        model = perform_model_training_steps(model, train_loader, num_steps)
    return get_params_grouped_by_algorithms(model)


def get_one_parameter_model_param_after_training_steps(model: nn.Module, num_steps=1):
    with set_torch_seed():
        train_loader = torch.utils.data.DataLoader(RandomDatasetMock(model.INPUT_SIZE),
                                                   batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        model = perform_model_training_steps(model, train_loader, num_steps)

    return model.parameters()


def test_if_algorithms_add_params(ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]):
    base_config, config_with_multiplier = ref_and_target_configs

    algo_to_params = get_lenet_params_after_training_steps(config_with_multiplier, num_steps=0)
    algo_names = get_multipliers_from_config(base_config).keys()

    assert sorted(algo_to_params.keys()) == sorted(list(algo_names) + ['regular'])


@pytest.mark.parametrize('creation_params', get_one_parameter_model_creation_params(for_training=False))
def test_if_parameter_is_initialized_correctly(creation_params: Dict):
    ref_model = create_initialized_one_parameter_model(nn.Parameter, **creation_params)
    target_model = create_initialized_one_parameter_model(CompressionParameter, **creation_params)

    assert pytest.approx(ref_model.param.data) == target_model.param.data
    assert ref_model.param.requires_grad == target_model.param.requires_grad

    if ref_model.param.requires_grad:
        get_one_parameter_model_param_after_training_steps(target_model)
    else:
        with pytest.raises(Exception):
            get_one_parameter_model_param_after_training_steps(target_model)


def check_if_grads_are_multiplied(ref_params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter],
                                  multiplier: float):
    ref_grads = get_grads(ref_params)
    ref_grads = [multiplier * grad for grad in ref_grads]
    target_grads = get_grads(target_params)

    check_equal(ref_grads, target_grads)


def test_if_setting_multipliers_in_config_multiplies_grads_values(
        ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]
):
    base_config, config_with_multiplier = ref_and_target_configs

    ref_params = get_lenet_params_after_training_steps(base_config)
    target_params = get_lenet_params_after_training_steps(config_with_multiplier)
    multipliers = get_multipliers_from_config(config_with_multiplier)
    multipliers['regular'] = 1

    for algo in ref_params:
        check_if_grads_are_multiplied(ref_params[algo], target_params[algo], multipliers[algo])


@pytest.mark.parametrize('creation_params', get_one_parameter_model_creation_params(for_training=True))
def test_if_setting_multiplier_in_parameter_multiplies_grads_values(creation_params: Dict):
    ref_model = create_initialized_one_parameter_model(nn.Parameter, **creation_params)
    target_model = create_initialized_one_parameter_model(CompressionParameter, **creation_params)

    ref_model_params = get_one_parameter_model_param_after_training_steps(ref_model)
    target_model_params = get_one_parameter_model_param_after_training_steps(target_model)

    assert list(target_model_params)[0].requires_grad
    check_if_grads_are_multiplied(ref_model_params, target_model_params, creation_params['multiplier'])


def check_if_zero_multiplier_freezes_training(params: Iterable[nn.Parameter], orig_params: Iterable[nn.Parameter],
                                              multiplier: float):
    if multiplier == 0:
        check_equal(orig_params, params)
    else:
        check_not_equal(orig_params, params)


def test_if_setting_multipliers_in_config_affect_train_speed(
        ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]
):
    _base_config, config_with_multiplier = ref_and_target_configs

    orig_params = get_lenet_params_after_training_steps(config_with_multiplier, num_steps=0)
    params = get_lenet_params_after_training_steps(config_with_multiplier, num_steps=1)
    multipliers = get_multipliers_from_config(config_with_multiplier)
    multipliers['regular'] = 1

    for algo in orig_params:
        check_if_zero_multiplier_freezes_training(params[algo], orig_params[algo], multipliers[algo])


def get_diff(params: Iterable[nn.Parameter], orig_params: Iterable[nn.Parameter]) -> List[torch.Tensor]:
    param_diffs = []
    for param, orig_param in zip(params, orig_params):
        param_diffs.append((param - orig_param).abs())
    return param_diffs


def check_params_affect_train_speed(orig_params: Iterable[nn.Parameter],
                                    params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter],
                                    compression_lr_multiplier: float):
    ref_diff = get_diff(ref_params, orig_params)
    target_diff = get_diff(params, orig_params)

    if pytest.approx(compression_lr_multiplier) == 1:
        check_equal(target_diff, ref_diff)
    elif compression_lr_multiplier > 1:
        check_less(target_diff, ref_diff)
    else:
        check_greater(target_diff, ref_diff)


@pytest.mark.parametrize('creation_params', get_one_parameter_model_creation_params(for_training=True))
def test_if_setting_multiplier_in_parameter_affect_train_speed(creation_params: Dict):
    ref_model = create_initialized_one_parameter_model(nn.Parameter, **creation_params)
    target_model = create_initialized_one_parameter_model(CompressionParameter, **creation_params)

    orig_model_params = list(ref_model.parameters())
    ref_model_params = get_one_parameter_model_param_after_training_steps(ref_model)
    target_model_params = get_one_parameter_model_param_after_training_steps(target_model)

    assert list(target_model_params)[0].requires_grad
    check_if_zero_multiplier_freezes_training(orig_model_params, target_model_params, creation_params['multiplier'])
    check_params_affect_train_speed(orig_model_params, ref_model_params, target_model_params,
                                    creation_params['multiplier'])
