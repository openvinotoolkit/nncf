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
from abc import ABC, abstractmethod
from typing import Dict
from typing import Callable
from typing import Any
from typing import Union
from typing import List
from typing import Tuple
from typing import TypeVar

import onnx
import numpy as np
import torch

from copy import deepcopy
from onnx import numpy_helper
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nncf.config.structures import BNAdaptationInitArgs
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.initialization import register_default_init_args
from nncf.torch.layers import NNCF_MODULES_MAP
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.initialization import PTInitializingDataLoader
from tests.common.command import Command as BaseCommand

TensorType = TypeVar('TensorType', bound=Union[torch.Tensor, np.ndarray])


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


def create_conv(in_channels, out_channels, kernel_size, weight_init=1, bias_init=0, padding=0, stride=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


def create_depthwise_conv(channels, kernel_size, weight_init, bias_init, padding=0, stride=1):
    conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, stride=stride, groups=channels)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


def create_transpose_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, stride):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


class BasicConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)
        self.wq_scale_shape_per_channel = (out_channels, 1, 1, 1)
        self.aq_scale_shape_per_channel = (1, in_channels, 1, 1)

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


class LeNet(nn.Module):
    INPUT_SIZE = 1, 32, 32

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_empty_config(model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None,
                     input_info: Dict = None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update({
        "model": "empty_config",
        "model_size": model_size,
        "input_info": input_info if input_info else _create_input_info()
    })
    return config


def get_grads(variables: List[nn.Parameter]) -> List[torch.Tensor]:
    return [var.grad.clone() for var in variables]


def to_numpy(tensor: TensorType) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor


def compare_tensor_lists(test: List[TensorType], reference: List[TensorType],
                         assert_fn: Callable[[np.ndarray, np.ndarray], bool]):
    assert len(test) == len(reference)

    for x, y in zip(test, reference):
        x = to_numpy(x)
        y = to_numpy(y)
        assert_fn(x, y)


def check_equal(test: List[TensorType], reference: List[TensorType], rtol: float = 1e-1):
    compare_tensor_lists(test, reference,
                         lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol))


def check_not_equal(test: List[TensorType], reference: List[TensorType], rtol: float = 1e-4):
    compare_tensor_lists(test, reference,
                         lambda x, y: np.testing.assert_raises(AssertionError,
                                                               np.testing.assert_allclose, x, y, rtol=rtol))


def check_less(test: List[TensorType], reference: List[TensorType], rtol=1e-4):
    check_not_equal(test, reference, rtol=rtol)
    compare_tensor_lists(test, reference, np.testing.assert_array_less)


def check_greater(test: List[TensorType], reference: List[TensorType], rtol=1e-4):
    check_not_equal(test, reference, rtol=rtol)
    compare_tensor_lists(test, reference,
                         lambda x, y: np.testing.assert_raises(AssertionError, np.testing.assert_array_less, x, y))


def create_compressed_model_and_algo_for_test(model: Module, config: NNCFConfig,
                                              dummy_forward_fn: Callable[[Module], Any] = None,
                                              wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
                                              resuming_state_dict: dict = None) \
        -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    assert isinstance(config, NNCFConfig)
    NNCFConfig.validate(config)
    algo, model = create_compressed_model(model, config, dump_graphs=False, dummy_forward_fn=dummy_forward_fn,
                                          wrap_inputs_fn=wrap_inputs_fn,
                                          resuming_state_dict=resuming_state_dict)
    return model, algo


def create_nncf_model_and_algo_builder(model: Module, config: NNCFConfig,
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
    composite_builder = PTCompositeCompressionAlgorithmBuilder(config, should_init=should_init)
    return compressed_model, composite_builder


def create_initialized_compressed_model(model: nn.Module, config: NNCFConfig, train_loader: DataLoader) -> nn.Module:
    config = register_default_init_args(deepcopy(config), train_loader, nn.MSELoss)
    model, _compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return model


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


class BaseDatasetMock(Dataset, ABC):
    def __init__(self, input_size: Tuple, num_samples: int = 10):
        super().__init__()
        self._input_size = input_size
        self._len = num_samples

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self) -> int:
        return self._len


class OnesDatasetMock(BaseDatasetMock):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(self._input_size), torch.ones(1)


class RandomDatasetMock(BaseDatasetMock):
    def __getitem__(self, index):
        return torch.rand(self._input_size), torch.zeros(1)


def create_any_mock_dataloader(dataset_cls: type, config: NNCFConfig, num_samples: int = 1) -> DataLoader:
    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = DataLoader(dataset_cls(input_sample_size[1:], num_samples),
                             batch_size=1,
                             num_workers=0,  # Workaround
                             shuffle=False, drop_last=True)
    return data_loader


def create_ones_mock_dataloader(config: NNCFConfig, num_samples: int = 1) -> DataLoader:
    return create_any_mock_dataloader(OnesDatasetMock, config, num_samples)


def create_random_mock_dataloader(config: NNCFConfig, num_samples: int = 1) -> DataLoader:
    return create_any_mock_dataloader(RandomDatasetMock, config, num_samples)


# ONNX graph helpers
def get_nodes_by_type(onnx_model: onnx.ModelProto, node_type: str) -> List[onnx.NodeProto]:
    retval = []
    for node in onnx_model.graph.node:
        if str(node.op_type) == node_type:
            retval.append(node)
    return retval


def get_all_inputs_for_graph_node(node: onnx.NodeProto, graph: onnx.GraphProto) -> \
        Dict[str, onnx.AttributeProto]:
    retval = {}
    for input_ in node.input:
        constant_input_nodes = [x for x in graph.node if input_ in x.output and x.op_type == "Constant"]
        for x in graph.initializer:
            if input_ == x.name:
                weight_tensor = x
                retval[input_] = numpy_helper.to_array(weight_tensor)
                # Only one weight tensor could be
                break

        for constant_input_node in constant_input_nodes:
            assert len(constant_input_node.attribute) == 1
            val = constant_input_node.attribute[0]
            retval[input_] = numpy_helper.to_array(val.t)

    return retval


class Command(BaseCommand):
    def run(self, timeout=3600, assert_returncode_zero=True):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # See runs_subprocess_in_precommit for more info on why this is needed
        return super().run(timeout, assert_returncode_zero)


def module_scope_from_node_name(name):
    module_name = name.rsplit('/', 1)[0].split(' ', 1)[1]
    return Scope.from_str(module_name)


class DummyDataLoader(PTInitializingDataLoader):
    def __init__(self):
        super().__init__([])

    @property
    def batch_size(self):
        return 1


def register_bn_adaptation_init_args(config: NNCFConfig):
    config.register_extra_structs([BNAdaptationInitArgs(data_loader=DummyDataLoader(), device=None)])
