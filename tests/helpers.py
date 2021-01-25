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
from typing import Dict, Callable, Any, Union, List
from typing import Tuple

import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch.nn import Module

from nncf.compression_method_api import CompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.layers import NNCF_MODULES_MAP
from nncf.model_creation import create_compressed_model, create_compression_algorithm_builders
from nncf.nncf_network import NNCFNetwork
from nncf.utils import get_all_modules_by_type


def fill_conv_weight(conv, value):
    conv.weight.data.fill_(value)
    with torch.no_grad():
        mask = torch.eye(conv.kernel_size[0])
        conv.weight += mask


def fill_bias(module, value):
    module.bias.data.fill_(value)


def fill_linear_weight(linear, value):
    linear.weight.data.fill_(value)
    with torch.no_grad():
        n = min(linear.in_features, linear.out_features)
        linear.weight[:n, :n] += torch.eye(n)


def create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, padding=0, stride=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


def create_transpose_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, stride):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


class BasicConvTestModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0., -1.],
                               [-1., 0.]]], [[[0., -1.],
                                              [-1., 0.]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2., -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size ** 2

    @property
    def bias_num(self):
        return self.kernel_size

    @property
    def nz_weights_num(self):
        return self.kernel_size * self.out_channels

    @property
    def nz_bias_num(self):
        return self.kernel_size


class TwoConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(1, 2, 2, -1, -2)))
        self.features.append(nn.Sequential(create_conv(2, 1, 3, 0, 0)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    @property
    def weights_num(self):
        return 8 + 18

    @property
    def bias_num(self):
        return 2 + 1

    @property
    def nz_weights_num(self):
        return 4 + 6

    @property
    def nz_bias_num(self):
        return 2


def get_empty_config(model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None,
                     input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info": input_info if input_info else _create_input_info()
    })
    return config


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def check_equal(test, reference, rtol=1e-4):
    for i, (x, y) in enumerate(zip(test, reference)):
        x = x.cpu().detach().numpy()
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg="Index: {}".format(i))


def create_compressed_model_and_algo_for_test(model: NNCFNetwork, config: NNCFConfig,
                                              dummy_forward_fn: Callable[[Module], Any] = None,
                                              wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                                              resuming_state_dict: dict = None) \
        -> Tuple[NNCFNetwork, CompressionAlgorithmController]:
    assert isinstance(config, NNCFConfig)
    NNCFConfig.validate(config)
    algo, model = create_compressed_model(model, config, dump_graphs=False, dummy_forward_fn=dummy_forward_fn,
                                          wrap_inputs_fn=wrap_inputs_fn,
                                          resuming_state_dict=resuming_state_dict)
    return model, algo


def create_nncf_model_and_algo_builder(model: NNCFNetwork, config: NNCFConfig,
                                       dummy_forward_fn: Callable[[Module], Any] = None,
                                       wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                                       resuming_state_dict: dict = None):
    assert isinstance(config, NNCFConfig)
    NNCFConfig.validate(config)
    input_info_list = create_input_infos(config)
    scopes_without_shape_matching = config.get('scopes_without_shape_matching', [])
    ignored_scopes = config.get('ignored_scopes')
    target_scopes = config.get('target_scopes')

    compressed_model = NNCFNetwork(model, input_infos=input_info_list,
                                   dummy_forward_fn=dummy_forward_fn,
                                   wrap_inputs_fn=wrap_inputs_fn,
                                   ignored_scopes=ignored_scopes,
                                   target_scopes=target_scopes,
                                   scopes_without_shape_matching=scopes_without_shape_matching)

    should_init = resuming_state_dict is None
    compression_algo_builder_list = create_compression_algorithm_builders(config, should_init=should_init)
    return compressed_model, compression_algo_builder_list


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.field = nn.Linear(1, 1)

    def forward(self, *input_, **kwargs):
        return None


def check_correct_nncf_modules_replacement(model: NNCFNetwork, compressed_model: NNCFNetwork) \
        -> Tuple[Dict[Scope, Module], Dict[Scope, Module]]:
    """
    Check that all convolutions in model was replaced by NNCF convolution.
    :param model: original model
    :param compressed_model: compressed model
    :return: list of all convolutions in  original model and list of all NNCF convolutions from compressed model
    """
    NNCF_MODULES_REVERSED_MAP = {value: key for key, value in NNCF_MODULES_MAP.items()}
    original_modules = get_all_modules_by_type(model, list(NNCF_MODULES_MAP.values()))
    nncf_modules = get_all_modules_by_type(compressed_model.get_nncf_wrapped_model(),
                                           list(NNCF_MODULES_MAP.keys()))
    assert len(original_modules) == len(nncf_modules)
    print(original_modules, nncf_modules)
    for scope in original_modules.keys():
        sparse_scope = deepcopy(scope)
        elt = sparse_scope.pop()  # type: ScopeElement
        elt.calling_module_class_name = NNCF_MODULES_REVERSED_MAP[elt.calling_module_class_name]
        sparse_scope.push(elt)
        print(sparse_scope, nncf_modules)
        assert sparse_scope in nncf_modules
    return original_modules, nncf_modules


class OnesDatasetMock:
    def __init__(self, input_size, num_samples=1):
        self.input_size = input_size
        super().__init__()
        self._len = num_samples

    def __getitem__(self, index):
        return torch.ones(self.input_size), torch.ones(1)

    def __len__(self):
        return self._len


def create_mock_dataloader(config, num_samples=1):
    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = torch.utils.data.DataLoader(OnesDatasetMock(input_sample_size[1:], num_samples),
                                              batch_size=1,
                                              num_workers=0,  # Workaround
                                              shuffle=False, drop_last=True)
    return data_loader
