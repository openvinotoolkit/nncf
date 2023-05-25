# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from nncf import nncf_logger
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging.logger import NNCFDeprecationWarning
from nncf.torch import register_module
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.layers import NNCFConv2d
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTInsertionType
from tests.torch.composite.test_sparsity_quantization import get_basic_sparsity_plus_quantization_config
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_models.synthetic import ManyNonEvalModules


@pytest.fixture()
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


def test_disable_shape_matching():
    class MatMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones([1]))

        def forward(self, inputs):
            half1, half2 = torch.chunk(inputs, 2, dim=2)
            return torch.bmm(half1, half2.transpose(1, 2))

    model = MatMulModel()

    input_shape_1 = (3, 32, 32)
    input_shape_2 = (4, 64, 64)

    qnet_no_shape = NNCFNetwork(
        deepcopy(model),
        input_infos=[
            ModelInputInfo(input_shape_1),
        ],
        scopes_without_shape_matching=["MatMulModel"],
    )  # type: NNCFNetwork

    context = qnet_no_shape.nncf.get_tracing_context()
    context.enable_trace_dynamic_graph()
    _ = qnet_no_shape(torch.zeros(*input_shape_1))
    graph_1 = deepcopy(qnet_no_shape.nncf.get_dynamic_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.nncf.get_dynamic_graph())

    assert graph_1 == graph_2

    nodes_1 = list(graph_1.get_all_nodes())
    assert len(nodes_1) == 5  # 1 input node + 1 chunk + 1 transpose + 1 matmul + 1 output node

    qnet = NNCFNetwork(
        model,
        input_infos=[
            ModelInputInfo(input_shape_1),
        ],
    )  # type: NNCFNetwork
    context = qnet.nncf.get_tracing_context()
    context.enable_trace_dynamic_graph()
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.nncf.get_dynamic_graph().get_nodes_count() > graph_1.get_nodes_count()


def test_check_correct_modules_replacement():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(TwoConvTestModel(), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    _, detected_nncf_modules = check_correct_nncf_modules_replacement(model, nncf_model)
    replaced_modules_reported_by_nncf_network = {
        scope: module for module, scope in nncf_model.nncf.get_nncf_modules().items()
    }
    assert set(detected_nncf_modules) == set(replaced_modules_reported_by_nncf_network)


class WeightNormedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(torch.nn.Conv1d(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


def test_weight_normed_modules_are_replaced_correctly():
    nncf_model = NNCFNetwork(WeightNormedConvModel(), input_infos=[ModelInputInfo([1, 1, 10])])

    wrapped_conv = nncf_model.conv
    assert hasattr(wrapped_conv, "weight_g")
    assert hasattr(wrapped_conv, "weight_v")
    assert hasattr(wrapped_conv, "weight")

    assert isinstance(wrapped_conv.weight_g, torch.nn.Parameter)
    assert isinstance(wrapped_conv.weight_v, torch.nn.Parameter)
    assert not isinstance(wrapped_conv.weight, torch.nn.Parameter)

    # pylint:disable=protected-access
    assert len(wrapped_conv._forward_pre_hooks) == 1


class UnregisteredUserModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1]))
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, input_):
        x = input_ * self.weight
        x += torch.rand_like(x)
        x = F.conv2d(x, self.conv.weight)
        x = self.conv(x)
        return x


@register_module()
class RegisteredUserModule(UnregisteredUserModule):
    pass


class TwoConvTestModelWithUserModule(TwoConvTestModel):
    def __init__(self):
        super().__init__()
        self.unregistered_user_module = UnregisteredUserModule()
        self.registered_user_module = RegisteredUserModule()

    def forward(self, x):
        x = super().forward(x)
        x = self.unregistered_user_module(x)
        x = self.registered_user_module(x)
        return x


def test_custom_module_registering():
    model = TwoConvTestModelWithUserModule()
    nncf_model = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    from nncf.torch.layers import UNWRAPPED_USER_MODULES

    assert RegisteredUserModule in UNWRAPPED_USER_MODULES.registry_dict.values()
    assert UnregisteredUserModule not in UNWRAPPED_USER_MODULES.registry_dict.values()

    # pylint: disable=protected-access
    modules = [nncf_model.registered_user_module, nncf_model.unregistered_user_module.conv]
    base_modules = [RegisteredUserModule, torch.nn.Conv2d]
    names = ["NNCFUserRegisteredUserModule", "NNCFConv2d"]
    for module, base_module, name in zip(modules, base_modules, names):
        assert isinstance(module, base_module)
        assert isinstance(module, _NNCFModuleMixin)
        assert type(module).__name__ == name

        module_attrs = dir(module)
        for attr in dir(_NNCFModuleMixin):
            assert attr in module_attrs

    # Check user ops metatypes
    graph = nncf_model.nncf.get_original_graph()
    nodes_dict = {
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/rand_like_0": UnknownMetatype,
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/conv2d_0": PTConv2dMetatype,
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/NNCFConv2d[conv]/conv2d_0": (
            PTModuleConv2dMetatype
        ),
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/rand_like_0": (
            UnknownMetatype
        ),
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/conv2d_0": (
            PTModuleConv2dMetatype
        ),
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/Conv2d[conv]/conv2d_0": (
            PTConv2dMetatype
        ),
    }
    for node_name, ref_metatype in nodes_dict.items():
        assert graph.get_node_by_name(node_name).metatype is ref_metatype


def test_get_weighted_original_graph_nodes():
    model = TwoConvTestModelWithUserModule()
    nncf_model = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    weighted_nodes = nncf_model.nncf.get_weighted_original_graph_nodes()
    ref_node_names = [
        "TwoConvTestModelWithUserModule/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModelWithUserModule/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/NNCFConv2d[conv]/conv2d_0",
        # The next one matched because it's a free op with metatypes corresponding to weighted ops
        # and is within a registered user module
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/conv2d_0",
    ]
    ref_weighted_nodes = [nncf_model.nncf.get_original_graph().get_node_by_name(name) for name in ref_node_names]
    assert set(weighted_nodes) == set(ref_weighted_nodes)


# pylint: disable=protected-access
def test_get_op_nodes_in_scope():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    nncf_graph = nncf_model.nncf.get_original_graph()

    # Valid scopes should be successfully found
    valid_nncf_modules = nncf_model.nncf.get_nncf_modules()
    nodes_list = list(nncf_graph.get_all_node_ids())
    for module_scope in valid_nncf_modules.values():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert len(matching_nncf_nodes) == 1
        node = matching_nncf_nodes[0]
        assert isinstance(node, NNCFNode)
        assert node.node_id in nodes_list

    fake_model = BasicConvTestModel()
    fake_nncf_model = NNCFNetwork(deepcopy(fake_model), input_infos=[ModelInputInfo([1, 1, 4, 4])])

    # Not valid scopes shouldn't be found
    fake_nncf_modules = fake_nncf_model.nncf.get_nncf_modules()
    for module_scope in fake_nncf_modules.values():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert not matching_nncf_nodes


def test_nncf_node_attrs_are_consistent():
    # Check that node returned from `add_nncf_node`
    # refer to the save `data` dict as node returned by
    # `get_node_by_id` and `get_op_nodes_in_scope`
    nncf_graph = PTNNCFGraph()
    new_node = nncf_graph.add_nncf_node(
        node_name="dummy", node_type="dummy", layer_name="dummy", node_metatype=UnknownMetatype
    )
    new_node_saved = nncf_graph.get_node_by_id(new_node.node_id)
    assert new_node.data is new_node_saved.data
    nodes_in_scope = nncf_graph.get_op_nodes_in_scope(nncf_graph.get_scope_by_node_name("dummy"))
    assert new_node.data is nodes_in_scope[0].data


def test_can_collect_scopes_of_train_only_modules():
    model = ManyNonEvalModules()
    graph_builder = GraphBuilder(custom_forward_fn=lambda model_: model_(torch.randn([1, 1, 1, 1])))
    graph = graph_builder.build_graph(model, as_eval=True)
    actual_scopes = [n.node_name for n in graph.get_all_nodes()]
    ref_scopes = {
        "ManyNonEvalModules/AvgPool2d[avg_pool]/avg_pool2d_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_1",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Linear[called_linear]/linear_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/CustomWeightModule[custom]/linear_0",
    }
    assert set(actual_scopes) == ref_scopes


def test_get_clean_shallow_copy():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    external_quantizers = getattr(sparse_quantized_model.nncf, EXTERNAL_QUANTIZERS_STORAGE_NAME)
    assert external_quantizers
    old_nncf_modules = sparse_quantized_model.nncf.get_nncf_modules()
    old_nncf_module_pre_ops = [module.pre_ops for module in old_nncf_modules]
    assert any(old_nncf_module_pre_ops)
    assert (
        sparse_quantized_model.nncf.get_graph().get_nodes_count()
        != sparse_quantized_model.nncf.get_original_graph().get_nodes_count()
    )

    old_interface = sparse_quantized_model.nncf
    clean_copy = sparse_quantized_model.nncf.get_clean_shallow_copy()
    assert clean_copy is sparse_quantized_model
    new_interface = clean_copy.nncf
    assert old_interface is not new_interface
    new_nncf_modules = clean_copy.nncf.get_nncf_modules()
    new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
    assert not any(new_nncf_module_pre_ops)
    assert clean_copy.nncf.get_graph().get_nodes_count() == clean_copy.nncf.get_original_graph().get_nodes_count()


class TwoConvTestModelWithUniqueFunction(TwoConvTestModel):
    def __init__(self):
        super().__init__()
        self.unique_attr = "unique_attr"
        self.non_unique_attr = "model_non_unique_attr"

    def train_step(self):
        pass

    @staticmethod
    def static_func():
        pass


def test_get_attr():
    model = TwoConvTestModelWithUniqueFunction()
    nncf_network = NNCFNetwork(model, [ModelInputInfo([1, 1, 4, 4])])

    assert hasattr(nncf_network, "unique_attr")
    assert hasattr(nncf_network, "non_unique_attr")
    assert inspect.ismethod(nncf_network.train_step)
    assert inspect.isfunction(nncf_network.static_func)


class ModelWithAttr(torch.nn.Module):
    CLASS_ATTR = 0

    def __init__(self):
        super().__init__()
        self.instance_attr = 0
        self._input_infos = 0  # deliberately set to coincide with an attr in NNCFNetwork

    def forward(self, x):
        return x

    def other_forward(self, x):
        return x * 2


def test_setting_attrs():
    model = ModelWithAttr()
    assert model.CLASS_ATTR == 0
    assert model.instance_attr == 0
    # pylint:disable=protected-access
    assert model._input_infos == 0
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])
    # pylint:disable=protected-access
    assert nncf_network._input_infos == 0

    nncf_network.instance_attr = 1
    assert nncf_network.instance_attr == 1

    nncf_network.non_original_model_attr = 1
    assert nncf_network.non_original_model_attr == 1
    assert hasattr(nncf_network, "non_original_model_attr")
    assert nncf_network.non_original_model_attr == 1


def mock_forward(*args, **kwargs):
    mock_forward.called = True


def mock_forward_with_self(self, *args, **kwargs):
    mock_forward_with_self.called = True


def mock_forward_with_matching_signature(self, x):
    mock_forward_with_matching_signature.called = True


@pytest.mark.parametrize("new_forward", [mock_forward, mock_forward_with_self, mock_forward_with_matching_signature])
def test_replacing_forward_with_free_functions(new_forward, _nncf_caplog):
    model = ModelWithAttr()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])

    nncf_network.forward = new_forward
    assert "set_original_unbound_forward" in _nncf_caplog.text


def test_replacing_forward_with_another_own_method(_nncf_caplog):
    model = ModelWithAttr()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])

    nncf_network.forward = nncf_network.other_forward
    assert "set_original_unbound_forward" in _nncf_caplog.text


def test_temporary_clean_view():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    old_sd = sparse_quantized_model.state_dict()
    old_graph = deepcopy(sparse_quantized_model.nncf.get_graph())
    with sparse_quantized_model.nncf.temporary_clean_view() as intermediate_model:
        clean_sd = intermediate_model.state_dict()
        assert len(clean_sd) < len(old_sd)
        new_nncf_modules = intermediate_model.nncf.get_nncf_modules()
        new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
        assert not any(new_nncf_module_pre_ops)
        assert (
            intermediate_model.nncf.get_graph().get_nodes_count()
            == intermediate_model.nncf.get_original_graph().get_nodes_count()
        )
    sd_after_tmp_clean_view = sparse_quantized_model.state_dict()
    for key in old_sd.keys():
        assert key in sd_after_tmp_clean_view  # pylint: disable=E1135
        assert torch.all(torch.eq(sd_after_tmp_clean_view[key], old_sd[key]))  # pylint: disable=E1136
    sparse_quantized_model.nncf.rebuild_graph()
    graph_after_tmp_clean_view = sparse_quantized_model.nncf.get_graph()
    assert graph_after_tmp_clean_view == old_graph


class MultipleForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, 1)
        self.conv1 = nn.Conv2d(1, 1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        return x1, x2


def test_multiple_forward():
    # Check that all convolution nodes in model have op_address and layer_attributes
    # for case with multiple forward of one module
    model = MultipleForwardModel()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    graph = sparse_quantized_model.nncf.get_original_graph()
    for node in list(graph.get_all_nodes())[1:-2]:
        assert node.layer_attributes is not None


def test_deepcopy_nncf_network():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    _ = deepcopy(sparse_quantized_model)


def test_insertion_point_target_point_translation():
    op_address = OperationAddress("dummy", Scope(), 0)
    for target_type in [PTInsertionType.NNCF_MODULE_POST_OP, TargetType.AFTER_LAYER]:
        with pytest.raises(RuntimeError):
            PTInsertionPoint(target_type, op_address)
    target_type = TargetType.POST_LAYER_OPERATION
    assert PTInsertionPoint(target_type, op_address).insertion_type == PTInsertionType.NNCF_MODULE_POST_OP


class IndirectModuleCaller(nn.Module):
    def __init__(self, module_for_indirection: torch.nn.Module):
        super().__init__()

        self.module_for_indirection = module_for_indirection
        self.conv_immediate = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def forward(self, img):
        enc_features = self.module_for_indirection.forward(img)
        out = self.conv_immediate(enc_features)

        return out


class Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # This is another source of indirection - a module whose reference then
        # is being passed to an owning container (Sequential). NNCFNetwork creation algo
        # must replace the module object in both places with the same object.
        self.conv_indirect = nn.Conv2d(self.in_channels, 32, 3, 2, 1, bias=False)
        self.features = nn.Sequential(self.conv_indirect, nn.BatchNorm2d(32), nn.ReLU6(inplace=True))

    def forward(self, img):
        enc_features = self.features(img)

        return enc_features


class ModelWithIndirectModuleCallBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.in_channels = in_channels

        self.backbone = Backbone(self.in_channels)
        self.testBranch = IndirectModuleCaller(self.backbone)

    def forward(self, img):
        out = self.testBranch(img)
        return out


def test_wrapping_of_indirect_module_operations():
    model = NNCFNetwork(ModelWithIndirectModuleCallBranch(), [ModelInputInfo([1, 3, 32, 32])])
    assert isinstance(model.backbone.conv_indirect, NNCFConv2d)
    assert model.backbone.features[0] is model.backbone.conv_indirect
    assert model.testBranch.module_for_indirection.features[0] is model.backbone.features[0]
    assert model.testBranch.module_for_indirection.conv_indirect is model.backbone.conv_indirect
    assert model.testBranch.module_for_indirection.features[0] is model.testBranch.module_for_indirection.conv_indirect
    assert isinstance(model.testBranch.conv_immediate, NNCFConv2d)


def test_can_work_with_sequential_models():
    sequential = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 1, 1))
    model = NNCFNetwork(sequential, [ModelInputInfo([1, 1, 32, 32])])
    assert model.nncf not in model  # Sequential, even wrapped, should only iterate over its real modules
    model(torch.ones([1, 1, 1, 1]))
    _ = model.nncf.get_clean_shallow_copy()


class SimplestModel(torch.nn.Module):
    INPUT_SIZE = [1, 1, 32, 32]

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(name="simple_net")
def simple_net_():
    model = NNCFNetwork(SimplestModel(), [ModelInputInfo(SimplestModel.INPUT_SIZE)])
    return model


def test_works_when_wrapped_with_dataparallel(simple_net):
    if not torch.cuda.is_available():
        pytest.xfail("The executing host must have > 1 CUDA GPU in order for this test to be relevant.")
    simple_net.cuda()
    dp_model = torch.nn.DataParallel(simple_net)
    dp_model(torch.zeros([10, *simple_net.INPUT_SIZE[1:]], device="cuda"))


def test_warns_on_old_style_calls(simple_net):
    with pytest.warns(NNCFDeprecationWarning):
        simple_net.get_graph()
    with pytest.warns(NNCFDeprecationWarning):
        simple_net.get_nncf_wrapped_model()


def test_class_has_same_name_and_module_as_original(simple_net):
    assert simple_net.__class__.__name__ == SimplestModel.__name__
    assert simple_net.__class__.__module__ == SimplestModel.__module__


def test_class_hashes_as_original(simple_net):
    assert hash(simple_net.__class__) == hash(SimplestModel)


def test_class_compares_as_original(simple_net):
    assert simple_net.__class__ == SimplestModel
    assert SimplestModel == simple_net.__class__
    assert simple_net.__class__ == simple_net.__class__
    assert not simple_net.__class__ != simple_net.__class__
    assert simple_net.__class__ != ModelWithAttr
    assert ModelWithAttr != simple_net.__class__


class MultiInputModel(torch.nn.Module):
    def forward(self, x, y):
        return x, y


def test_forward_signature_is_same_as_for_original_model(simple_net):
    original_obj = SimplestModel()
    original_signature_inst = inspect.signature(original_obj.forward)
    net_signature_inst = inspect.signature(simple_net.forward)
    assert original_signature_inst == net_signature_inst

    original_signature_cls = inspect.signature(SimplestModel.forward)
    net_signature_cls = inspect.signature(simple_net.__class__.forward)
    assert original_signature_cls == net_signature_cls

    # Verify that if we create 2 NNCFNetworks, then each will have its own signature
    another_original_obj = MultiInputModel()
    another_nncf_net = NNCFNetwork(
        MultiInputModel(), input_infos=[ModelInputInfo([1, 1, 1, 1]), ModelInputInfo([1, 1, 1, 1])]
    )
    assert inspect.signature(another_nncf_net.forward) == inspect.signature(another_original_obj.forward)
    assert inspect.signature(simple_net.forward) == inspect.signature(original_obj.forward)


class MetaModel(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def method(self):
        pass


class ConcreteMetaModel(MetaModel):
    def method(self):
        pass

    def forward(self, x):
        return x


def test_can_wrap_models_with_metaclass():
    _ = NNCFNetwork(ConcreteMetaModel(), [ModelInputInfo([1, 1, 1, 1])])


def test_reset_original_unbound_forward():
    model = ModelWithAttr()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])
    inp = torch.ones((1,))
    assert nncf_network.forward(inp) == inp

    nncf_network.set_original_unbound_forward(model.__class__.other_forward)
    assert nncf_network.forward(inp) == inp * 2

    nncf_network.reset_original_unbound_forward()
    assert nncf_network.forward(inp) == inp


def test_wrap_original_forward():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    def get_wrapper(fun):
        def wrapper(*args, **kwargs):
            new_args = (args[0], args[1] + 1, *args[2:])
            return fun(*new_args, **kwargs)

        return wrapper

    inp = torch.ones((1,))

    original_model = Model()
    assert str(inspect.signature(original_model.forward)) == "(x)"

    nncf_model = NNCFNetwork(deepcopy(original_model), [ModelInputInfo([1])])
    assert original_model.forward(inp) == nncf_model.forward(inp) == inp

    Model.forward = get_wrapper(Model.forward)
    assert str(inspect.signature(original_model.forward)) == "(*args, **kwargs)"

    assert inspect.signature(original_model.forward) != inspect.signature(nncf_model.forward)
    assert original_model.forward(inp) == nncf_model.forward(inp) == inp + 1

    # Ideally, the condition below should fail but currently there is no means to implement it
    assert inspect.signature(original_model.forward) != inspect.signature(nncf_model.forward)


def test_forward_hooks_are_preserved():
    # Testing only for the forward hook - other hooks use the same mechanism.
    original_obj = SimplestModel()

    class CallCounter:
        def __init__(self):
            self.enabled = True
            self.call_count = 0

        def __call__(self, *args, **kwargs):
            if self.enabled:
                self.call_count += 1

    hook = CallCounter()
    original_obj.register_forward_hook(hook)

    hook.enabled = False
    nncf_net = NNCFNetwork(original_obj, [ModelInputInfo(SimplestModel.INPUT_SIZE)])
    hook.enabled = True

    assert len(nncf_net._forward_hooks) == 1
    assert next(iter(nncf_net._forward_hooks.values())) is hook
    nncf_net(torch.ones(SimplestModel.INPUT_SIZE))
    assert hook.call_count == 1
