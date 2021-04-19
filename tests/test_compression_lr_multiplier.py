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

import pytest
import torch
from torch import nn
from torch.optim import SGD

from nncf.layer_utils import CompressionParameter
from nncf.layer_utils import CompressionParameter
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import check_equal
from tests.helpers import check_not_equal
from tests.helpers import get_grads
from tests.helpers import LeNet
from tests.quantization.test_algo_quantization import get_quantization_config_without_range_init
from tests.sparsity.rb.test_algo import get_basic_sparsity_config


@contextlib.contextmanager
def torch_seed():
    seed = torch.seed()
    torch.manual_seed(0)
    yield
    torch.manual_seed(seed)


def get_quantization_config():
    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE)
    return config


def get_sparsity_config():
    config = get_basic_sparsity_config([1, 1, LeNet.INPUT_SIZE, LeNet.INPUT_SIZE])
    return config


def modify_config(config, multiplier, multiplier_location, use_algo_list=False):
    config = copy.deepcopy(config)

    if use_algo_list:
        config['compression'] = [config['compression']]

    if multiplier_location in ['local', 'both']:
        algorithms = []
        if isinstance(config['compression'], list):
            algorithms = config['compression']
        else:
            algorithms.append(config['compression'])

        for algo in algorithms:
            algo.update({
                'compression_lr_multiplier': multiplier
            })

    if multiplier_location in ['global', 'both']:
        config.update({
            'compression_lr_multiplier': multiplier
        })

    return config


def divide_params(params):
    regular_params, compression_params = [], []
    for param in params:
        if param.requires_grad:
            if isinstance(param, CompressionParameter):
                compression_params.append(param)
            else:
                regular_params.append(param)

    return regular_params, compression_params


def make_train_steps(config, num_steps=1):
    with torch_seed():
        model = LeNet()
        model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        for param in model.parameters():
            if param.requires_grad:
                nn.init.normal_(param)
        optimizer = SGD(model.parameters(), lr=0.1)

        for i in range(num_steps):
            optimizer.zero_grad()
            x = torch.rand(config['input_info']['sample_size'])
            y = model(x)
            loss = (y ** 2).sum()

            loss.backward()
            optimizer.step()

    return model.parameters()


@pytest.mark.parametrize('get_config', [
    get_quantization_config,
    get_sparsity_config
])
@pytest.mark.parametrize('multiplier', [0, 1, 10])
@pytest.mark.parametrize('multiplier_location', ['local', 'global', 'both'])
@pytest.mark.parametrize('use_algo_list', [True, False])
def test_gradients(get_config, multiplier, multiplier_location, use_algo_list):
    base_config = get_config()
    config_with_multiplier = modify_config(base_config, multiplier, multiplier_location, use_algo_list)

    ref_params, ref_compression_params = divide_params(make_train_steps(base_config))
    target_params, target_compression_params = divide_params(make_train_steps(config_with_multiplier))
    ref_grads, ref_compression_grads = get_grads(ref_params), get_grads(ref_compression_params)
    target_grads, target_compression_grads = get_grads(target_params), get_grads(target_compression_params)

    ref_compression_grads = [multiplier * grad for grad in ref_compression_grads]
    check_equal(ref_grads, target_grads)
    check_equal(ref_compression_grads, target_compression_grads)


@pytest.mark.parametrize('get_config', [
    get_quantization_config,
    get_sparsity_config
])
@pytest.mark.parametrize('multiplier', [0, 1])
@pytest.mark.parametrize('multiplier_location', ['local', 'global', 'both'])
@pytest.mark.parametrize('use_algo_list', [True, False])
def test_parameters(get_config, multiplier, multiplier_location, use_algo_list):
    base_config = get_config()
    config_with_multiplier = modify_config(base_config, multiplier, multiplier_location, use_algo_list)

    orig_params, orig_compression_params = divide_params(make_train_steps(base_config, num_steps=0))
    params, compression_params = divide_params(make_train_steps(config_with_multiplier))

    if multiplier == 0:
        check_equal(orig_compression_params, compression_params)
    else:
        check_not_equal(orig_compression_params, compression_params)
    check_not_equal(orig_params, params)
