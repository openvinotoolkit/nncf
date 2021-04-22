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
from torch.optim import SGD

from nncf import NNCFConfig
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import check_equal
from tests.helpers import get_grads
from tests.helpers import LeNet
from tests.quantization.test_algo_quantization import get_quantization_config_without_range_init
from tests.sparsity.rb.test_algo import get_basic_sparsity_config


ALGO_NAME_TO_PATH_MAP = {
    'quantization': 'nncf.quantization',
    'rb_sparsity': 'nncf.sparsity.rb'
}


@contextlib.contextmanager
def set_torch_seed(seed=42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)


def get_quantization_config() -> NNCFConfig:
    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE)
    return config


def get_sparsity_config() -> NNCFConfig:
    config = get_basic_sparsity_config([1, 1, LeNet.INPUT_SIZE, LeNet.INPUT_SIZE])
    return config


def get_config_algorithms(config: NNCFConfig) -> List[Dict]:
    algorithms = []

    if isinstance(config['compression'], list):
        algorithms = config['compression']
    else:
        algorithms.append(config['compression'])

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
    get_orig_config_fns = [get_quantization_config, get_sparsity_config]
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
            'multipliers': multipliers,
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


def get_params_grouped_by_algorithms(model: nn.Module) -> Dict[str, Iterable[nn.Parameter]]:
    cls_name_to_params = {}
    modules = model.modules()
    for module in modules:
        params = module.parameters(recurse=False)
        full_cls_name = '.'.join([module.__class__.__module__, module.__class__.__name__])
        if full_cls_name not in cls_name_to_params:
            cls_name_to_params[full_cls_name] = []
        cls_name_to_params[full_cls_name].extend(params)

    algo_name_to_params = {'regular': []}
    for cls_name, params in cls_name_to_params.items():
        params = [param for param in params if param.requires_grad]
        for algo_name, algo_path in ALGO_NAME_TO_PATH_MAP.items():
            if algo_path in cls_name:
                if algo_name not in algo_name_to_params:
                    algo_name_to_params[algo_name] = []
                algo_name_to_params[algo_name].extend(params)
                break
        else:
            algo_name_to_params['regular'].extend(params)

    return algo_name_to_params


def get_params_after_train_steps(config: NNCFConfig, num_steps: int = 1) -> Dict[str, Iterable[nn.Parameter]]:
    with set_torch_seed():
        model = LeNet()
        model, _compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        for param in model.parameters():
            if param.requires_grad:
                nn.init.normal_(param)
        optimizer = SGD(model.parameters(), lr=0.1)

        for _ in range(num_steps):
            optimizer.zero_grad()
            x = torch.rand(config['input_info']['sample_size'])
            y = model(x)
            loss = (y ** 2).sum()

            loss.backward()
            optimizer.step()

    return get_params_grouped_by_algorithms(model)


def test_if_algorithms_add_params(ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]):
    base_config, config_with_multiplier = ref_and_target_configs

    algo_to_params = get_params_after_train_steps(config_with_multiplier)
    algo_names = get_multipliers_from_config(base_config).keys()

    assert sorted(algo_to_params.keys()) == sorted(list(algo_names) + ['regular'])


def test_how_multipliers_affect_grads(ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]):
    base_config, config_with_multiplier = ref_and_target_configs

    ref_params = get_params_after_train_steps(base_config)
    target_params = get_params_after_train_steps(config_with_multiplier)
    multipliers = get_multipliers_from_config(config_with_multiplier)
    multipliers['regular'] = 1

    for algo in multipliers:
        ref_grads = get_grads(ref_params[algo])
        ref_grads = [multipliers[algo] * grad for grad in ref_grads]
        target_grads = get_grads(target_params[algo])

        check_equal(ref_grads, target_grads)


def test_how_multipliers_affect_params_change(ref_and_target_configs: Tuple[NNCFConfig, NNCFConfig]):
    _base_config, config_with_multiplier = ref_and_target_configs

    orig_params = get_params_after_train_steps(config_with_multiplier, num_steps=0)
    params = get_params_after_train_steps(config_with_multiplier, num_steps=1)
    multipliers = get_multipliers_from_config(config_with_multiplier)
    multipliers['regular'] = 1

    for algo in multipliers:
        if multipliers[algo] == 0:
            check_equal(orig_params[algo], params[algo])
        else:
            with pytest.raises(AssertionError):
                check_equal(orig_params[algo], params[algo])
