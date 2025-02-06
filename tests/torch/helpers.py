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

import contextlib
import numbers
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.config import NNCFConfig
from nncf.config.extractors import extract_algorithm_names
from nncf.config.structures import BNAdaptationInitArgs
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.initialization import register_default_init_args
from nncf.torch.layer_utils import StatefullModuleInterface
from nncf.torch.layers import NNCF_MODULES_MAP
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_module_replacement import get_original_module_scope_from_nncf_module_scope
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_all_modules_by_type
from tests.cross_fw.shared.command import Command as BaseCommand
from tests.cross_fw.shared.comparator import BaseTensorListComparator

TensorType = Union[torch.Tensor, np.ndarray, numbers.Number]


def fill_conv_weight(conv, value, dim=2):
    conv.weight.data.fill_(value)
    # TODO: Fill it right
    if dim in [2, 3]:
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


def fill_params_of_model_by_normal(model, std=1.0):
    for param in model.parameters():
        param.data = torch.normal(0, std, size=param.data.size())


def create_conv(
    in_channels, out_channels, kernel_size, weight_init=1, bias_init=0, padding=0, stride=1, dim=2, bias=True
):
    conv_dim_map = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    conv = conv_dim_map[dim](in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
    fill_conv_weight(conv, weight_init, dim)
    if bias:
        fill_bias(conv, bias_init)

    return conv


def create_bn(num_features, dim=2):
    bn_dim_map = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    return bn_dim_map[dim](num_features)


def create_grouped_conv(
    in_channels, out_channels, kernel_size, groups, weight_init=1, bias_init=0, padding=0, stride=1
):
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise nncf.ValidationError(
            "Cannot create grouped convolution. Either `in_channels` or `out_channels` are not divisible by `groups`"
        )
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, padding=padding, stride=stride)
    fill_conv_weight(conv, weight_init)
    fill_bias(conv, bias_init)
    return conv


def create_depthwise_conv(channels, kernel_size, weight_init, bias_init, padding=0, stride=1, dim=2):
    conv_dim_map = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    conv = conv_dim_map[dim](channels, channels, kernel_size, padding=padding, stride=stride, groups=channels)
    fill_conv_weight(conv, weight_init, dim)
    fill_bias(conv, bias_init)
    return conv


def create_transpose_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, stride, dim=2):
    conv_dim_map = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
    conv = conv_dim_map[dim](in_channels, out_channels, kernel_size, stride=stride)
    fill_conv_weight(conv, weight_init, dim)
    fill_bias(conv, bias_init)
    return conv


class BasicConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self, in_channels=1, out_channels=2, kernel_size=2, weight_init=-1, bias_init=-2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.conv = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init, padding)
        self.wq_scale_shape_per_channel = (out_channels, 1, 1, 1)
        self.aq_scale_shape_per_channel = (1, in_channels, 1, 1)

    @staticmethod
    def default_weight():
        return torch.tensor([[[[0.0, -1.0], [-1.0, 0.0]]], [[[0.0, -1.0], [-1.0, 0.0]]]])

    @staticmethod
    def default_bias():
        return torch.tensor([-2.0, -2])

    def forward(self, x):
        return self.conv(x)

    @property
    def weights_num(self):
        return self.out_channels * self.kernel_size**2

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
    INPUT_SHAPE = [1, 1, 4, 4]
    NNCF_CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
    ]
    CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/Conv2d[0]/conv2d_0",
    ]

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


class TwoSharedConvTestModel(nn.Module):
    INPUT_SHAPE = [1, 1, 4, 4]
    NNCF_CONV_NODES_NAMES = [
        "TwoSharedConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoSharedConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
    ]
    CONV_NODES_NAMES = [
        "TwoSharedConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]/conv2d_0",
        "TwoSharedConvTestModel/Sequential[features]/Sequential[1]/Conv2d[0]/conv2d_0",
    ]

    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(1, 1, 1, -1, -2)))
        self.features.append(nn.Sequential(create_conv(1, 1, 1, 0, 0)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        for _ in range(2):
            x = self.features(x)
        return x


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


class DummyOpWithState(torch.nn.Module, StatefullModuleInterface):
    def __init__(self, state: str):
        super().__init__()
        self._state = state
        # Keep dummy param to check state dict
        self._dummy_param = torch.nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(self, *args):
        if len(args) == 1:
            return args[0] + self._dummy_param
        # To work correctly with
        # TargetType.PRE_LAYER_OPERATION
        # TargetType.POST_LAYER_OPERATION
        args[0].weight + self._dummy_param
        return None

    def get_config(self):
        return self._state

    @classmethod
    def from_config(cls, state: str):
        return cls(state)


def commands_are_equal(
    command_left: Union[PTInsertionCommand, PTSharedFnInsertionCommand],
    command_right: Union[PTInsertionCommand, PTSharedFnInsertionCommand],
    check_priority: bool = True,
    check_hooks_group_name: bool = True,
    check_fn_ref=True,
) -> bool:
    """
    Returns True if given commands are equal and False elsewhere.

    :param command_left: The first command.
    :param command_right: The second command.
    :param check_priority: Whether to check insertion priority or not.
    :param check_hooks_group_name: Whether to check hooks group name or not.
    :param check_fn_ref: Whether to check fn by reference or not.
    :returns: True if given commands are equal and False elsewhere.
    """
    if type(command_right) is not type(command_left):
        return False

    # Check reference to functions are equal.
    if check_fn_ref and command_right.fn is not command_left.fn:
        return False
    if check_hooks_group_name and command_right.hooks_group_name != command_left.hooks_group_name:
        return False
    if check_priority and command_right.priority != command_left.priority:
        return False

    if isinstance(command_right, PTInsertionCommand):
        if command_left.target_point != command_right.target_point:
            return False
    elif isinstance(command_right, PTSharedFnInsertionCommand):
        if not all(a == b for a, b in zip(command_left.target_points, command_right.target_points)):
            return False
        if (
            command_right.target_points != command_left.target_points
            or command_right.op_name != command_left.op_name
            or command_right.compression_module_type != command_left.compression_module_type
        ):
            return False
    else:
        raise RuntimeError()
    return True


class SharedConv(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 1, 2, 2)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv(x + 1)
        return a + b


class SharedCustomConv(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1, 2, 2)))

    def forward(self, x):
        a = F.conv2d(x, self.weight)
        b = F.conv2d(x + 1, self.weight)
        return a + b


def get_empty_config(
    model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None, input_info: Dict = None
) -> NNCFConfig:
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update(
        {
            "model": "empty_config",
            "model_size": model_size,
            "input_info": input_info if input_info else _create_input_info(),
        }
    )
    return config


def get_grads(variables: List[nn.Parameter]) -> List[torch.Tensor]:
    return [var.grad.clone() for var in variables]


class PTTensorListComparator(BaseTensorListComparator):
    @classmethod
    def _to_numpy(cls, tensor: TensorType) -> Union[np.ndarray, numbers.Number]:
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        if isinstance(tensor, (np.ndarray, numbers.Number)):
            return tensor
        raise Exception(f"Tensor must be np.ndarray or torch.Tensor, not {type(tensor)}")


def create_compressed_model_and_algo_for_test(
    model: Module,
    config: NNCFConfig = None,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
    compression_state: Dict[str, Any] = None,
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    if config is not None:
        assert isinstance(config, NNCFConfig)
        NNCFConfig.validate(config)
    algo, model = create_compressed_model(
        model,
        config,
        dump_graphs=False,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        compression_state=compression_state,
    )
    return model, algo


def create_nncf_model_and_single_algo_builder(
    model: Module,
    config: NNCFConfig,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    assert isinstance(config, NNCFConfig)
    NNCFConfig.validate(config)
    input_info = FillerInputInfo.from_nncf_config(config)
    scopes_without_shape_matching = config.get("scopes_without_shape_matching", [])
    ignored_scopes = config.get("ignored_scopes")
    target_scopes = config.get("target_scopes")

    compressed_model = NNCFNetwork(
        model,
        input_info=input_info,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        ignored_scopes=ignored_scopes,
        target_scopes=target_scopes,
        scopes_without_shape_matching=scopes_without_shape_matching,
    )

    algo_names = extract_algorithm_names(config)
    assert len(algo_names) == 1
    algo_name = next(iter(algo_names))
    builder_cls = PT_COMPRESSION_ALGORITHMS.get(algo_name)
    builder = builder_cls(config, should_init=True)
    return compressed_model, builder


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


class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input_, **kwargs):
        return None


class ModelWithReloadedForward(nn.Module):
    """
    Model accepts tensor or a dict in format
    {"tensor": input_tensor}
    """

    INPUT_SHAPE = [1, 1]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        if isinstance(x, dict):
            self.forward(x["tensor"])
        return self.linear(x)


def check_correct_nncf_modules_replacement(
    model: torch.nn.Module, compressed_model: NNCFNetwork
) -> Tuple[Dict[Scope, Module], Dict[Scope, Module]]:
    """
    Checks that all extendable modules in model were replaced by NNCF-extended counterparts.
    :param model: original model
    :param compressed_model: compressed model
    :return: list of all extendable modules in `model` and list of all NNCF-extended modules
     in `compressed_model`
    """
    original_modules = get_all_modules_by_type(model, list(NNCF_MODULES_MAP.values()))
    nncf_modules = get_all_modules_by_type(compressed_model, list(NNCF_MODULES_MAP.keys()))
    assert len(original_modules) == len(nncf_modules)
    for nncf_scope in nncf_modules:
        original_scope = get_original_module_scope_from_nncf_module_scope(nncf_scope)
        assert original_scope in original_modules
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


def create_any_mock_dataloader(
    dataset_cls: type, config: NNCFConfig, num_samples: int = 1, batch_size: int = 1
) -> DataLoader:
    input_info = FillerInputInfo.from_nncf_config(config)
    input_sample_size = input_info.elements[0].shape
    data_loader = DataLoader(
        dataset_cls(input_sample_size[1:], num_samples),
        batch_size=batch_size,
        num_workers=0,  # Workaround
        shuffle=False,
        drop_last=True,
    )
    return data_loader


def create_ones_mock_dataloader(config: NNCFConfig, num_samples: int = 1, batch_size: int = 1) -> DataLoader:
    return create_any_mock_dataloader(OnesDatasetMock, config, num_samples, batch_size)


def create_random_mock_dataloader(config: NNCFConfig, num_samples: int = 1, batch_size: int = 1) -> DataLoader:
    return create_any_mock_dataloader(RandomDatasetMock, config, num_samples, batch_size)


# ONNX graph helpers
def get_nodes_by_type(onnx_model: onnx.ModelProto, node_type: str) -> List[onnx.NodeProto]:
    retval = []
    for node in onnx_model.graph.node:
        if str(node.op_type) == node_type:
            retval.append(node)
    return retval


def get_all_inputs_for_graph_node(node: onnx.NodeProto, graph: onnx.GraphProto) -> Dict[str, onnx.AttributeProto]:
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


def resolve_constant_node_inputs_to_values(
    node: onnx.NodeProto, graph: onnx.GraphProto
) -> Dict[str, onnx.AttributeProto]:
    retval = {}
    for input_ in node.input:
        constant_input_nodes = [x for x in graph.node if input_ in x.output and x.op_type == "Constant"]
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
    module_name = name.rsplit("/", 1)[0].split(" ", 1)[1]
    return Scope.from_str(module_name)


class DummyDataLoader(PTInitializingDataLoader):
    def __init__(self):
        super().__init__([])

    @property
    def batch_size(self):
        return 1


def register_bn_adaptation_init_args(config: NNCFConfig):
    config.register_extra_structs([BNAdaptationInitArgs(data_loader=DummyDataLoader(), device=None)])


@contextlib.contextmanager
def set_torch_seed(seed: int = 42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)


def create_dataloader_with_num_workers(create_dataloader, num_workers, sample_type):
    def create_dataloader_classification(*args, **kwargs):
        train_loader, train_sampler, val_loader, init_loader = create_dataloader(*args, **kwargs)
        init_loader.num_workers = num_workers
        return train_loader, train_sampler, val_loader, init_loader

    def create_dataloader_semantic_segmentation(*args, **kwargs):
        (train_loader, val_loader, init_loader), class_weights = create_dataloader(*args, **kwargs)
        init_loader.num_workers = num_workers
        return (train_loader, val_loader, init_loader), class_weights

    def create_dataloader_object_detection(*args, **kwargs):
        test_data_loader, train_data_loader, init_data_loader = create_dataloader(*args, **kwargs)
        init_data_loader.num_workers = num_workers
        return test_data_loader, train_data_loader, init_data_loader

    if sample_type == "classification":
        return create_dataloader_classification
    if sample_type == "semantic_segmentation":
        return create_dataloader_semantic_segmentation
    if sample_type == "object_detection":
        return create_dataloader_object_detection


def load_exported_onnx_version(
    nncf_config: NNCFConfig, model: torch.nn.Module, path_to_storage_dir: Path, save_format: str = None
) -> onnx.ModelProto:
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    onnx_checkpoint_path = path_to_storage_dir / "model.onnx"
    compression_ctrl.export_model(str(onnx_checkpoint_path), save_format=save_format)
    model_proto = onnx.load_model(str(onnx_checkpoint_path))
    return model_proto


HookType = TypeVar("HookType")


class HookChecker:
    """
    Class to check pre/post hooks and pre ops are placed correctly.
    Supports check for one wrapped NNCFModule for now.
    """

    def __init__(self, target_model: torch.nn.Module, nncf_module_attr_name: str):
        """
        :param nncf_module_attr_name: name of the nncf module attribute name in target model.
        """
        self._nncf_module_attr_name = nncf_module_attr_name
        self._target_model = target_model
        self._ref_hooks = defaultdict(dict)

    def add_ref(
        self,
        ref_hooks: List[callable],
        target_type: TargetType,
        target_node_name: str,
        input_port_id: int,
    ) -> None:
        """
        Adds references hooks.
        """
        op_address = self._convert_to_op_address(
            target_type, target_node_name, input_port_id, self._target_model.nncf.replace_modules
        )
        self._ref_hooks[target_type].update({op_address: ref_hooks})

    def _convert_to_op_address(
        self, target_type: TargetType, target_node_name: str, input_port_id: int, replace_modules: bool
    ) -> Any:
        address_map = self._target_model.nncf.get_node_to_op_address_mapping()
        address = address_map[target_node_name]
        if replace_modules:
            if target_type == TargetType.OPERATOR_PRE_HOOK:
                address = PreHookId(address, input_port_id)
            elif target_type in [
                TargetType.OPERATION_WITH_WEIGHTS,
                TargetType.PRE_LAYER_OPERATION,
                TargetType.POST_LAYER_OPERATION,
            ]:
                address = getattr(self._target_model, self._nncf_module_attr_name)
        else:
            if target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATION_WITH_WEIGHTS]:
                address = PreHookId(address, input_port_id)
            elif target_type in [
                TargetType.PRE_LAYER_OPERATION,
                TargetType.POST_LAYER_OPERATION,
            ]:
                address = getattr(self._target_model, self._nncf_module_attr_name)
        return address

    def check_with_reference(self):
        """
        Check hooks in the target model and reference hooks are matching.
        """
        self._check_weight_update_hooks(self._ref_hooks[TargetType.OPERATION_WITH_WEIGHTS])

        target_module = getattr(self._target_model, self._nncf_module_attr_name)
        if target_module in self._ref_hooks[TargetType.PRE_LAYER_OPERATION]:
            hooks = target_module.pre_ops
            self._check_pre_post_op_hooks(hooks, self._ref_hooks[TargetType.PRE_LAYER_OPERATION][target_module])
        if target_module in self._ref_hooks[TargetType.POST_LAYER_OPERATION]:
            hooks = target_module.post_ops
            self._check_pre_post_op_hooks(hooks, self._ref_hooks[TargetType.POST_LAYER_OPERATION][target_module])

        hooks = self._target_model.nncf._compressed_context._pre_hooks
        self._check_pre_post_hooks(hooks, self._ref_hooks[TargetType.OPERATOR_PRE_HOOK])
        hooks = self._target_model.nncf._compressed_context._post_hooks
        self._check_pre_post_hooks(hooks, self._ref_hooks[TargetType.OPERATOR_POST_HOOK])

    def clear(self):
        """
        Removes all recorded references.
        """
        self._ref_hooks.clear()

    @staticmethod
    def _check_weight_update_hooks(ref_hooks: Dict[torch.nn.Module, List[HookType]]):
        for target_module, ref_hooks_per_module in ref_hooks.items():
            assert len(target_module.pre_ops) == len(ref_hooks_per_module)
            for actual_op, ref_op in zip(target_module.pre_ops.values(), ref_hooks_per_module):
                assert isinstance(actual_op, UpdateWeight)
                assert actual_op.op is ref_op

    @staticmethod
    def _check_pre_post_op_hooks(hooks: List[torch.ModuleDict], ref_hooks: List[HookType]):
        assert len(hooks) == len(ref_hooks)
        for actual_hook, ref_hook in zip(hooks.values(), ref_hooks):
            assert actual_hook is ref_hook

    @staticmethod
    def _check_pre_post_hooks(
        hooks: Dict[OperationAddress, Dict[Any, HookType]], ref_hooks: Dict[OperationAddress, List[HookType]]
    ):
        assert len(hooks) == len(ref_hooks)
        for op_address, ref_hooks in ref_hooks.items():
            actual_hooks = hooks[op_address].values()
            assert len(actual_hooks) == len(ref_hooks)
            for actual_hook, ref_hook in zip(actual_hooks, ref_hooks):
                assert actual_hook is ref_hook
