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
from collections import Counter
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from copy import deepcopy
from torchvision.models import resnet50

from examples.common.models.classification import squeezenet1_1_custom
from nncf.checkpoint_loading import load_state
from nncf.compression_method_api import CompressionLoss, CompressionScheduler
from nncf.dynamic_graph.context import ScopeElement, Scope
from nncf.dynamic_graph.graph import OperationExecutionContext, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.trace_tensor import TensorMeta
from nncf.hw_config import HWConfigType
from nncf.layers import NNCFConv2d
from nncf.model_creation import create_compression_algorithm_builders
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_network import CompressionModuleType, InsertionInfo
from nncf.quantization.algo import QuantizationController, QuantizationBuilder
from nncf.quantization.layers import QuantizationMode, QuantizerConfig, SymmetricQuantizer, BaseQuantizer, \
    QUANTIZATION_MODULES
from nncf.quantization.quantizer_id import NonWeightQuantizerId
from nncf.utils import get_all_modules_by_type
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init, \
    get_squeezenet_quantization_config
from tests.helpers import BasicConvTestModel, TwoConvTestModel, get_empty_config, \
    create_compressed_model_and_algo_for_test, MockModel, create_conv


def compare_qconfigs(config: QuantizerConfig, quantizer: BaseQuantizer):
    assert config.is_weights == quantizer.is_weights
    assert config.bits == quantizer.num_bits
    assert isinstance(quantizer, QUANTIZATION_MODULES.get(config.mode))
    assert config.per_channel == quantizer.per_channel
    assert config.signedness_to_force == quantizer.signedness_to_force


def test_quantization_configs__with_defaults():
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qconfig = QuantizerConfig(8, QuantizationMode.SYMMETRIC, None, False, None, True)
    for wq in weight_quantizers.values():
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(8, QuantizationMode.SYMMETRIC, None, False, None, False)
    for aq_info in activation_quantizer_infos.values():
        compare_qconfigs(ref_activation_qconfig, aq_info.quantizer_module_ref)


def test_quantization_configs__custom():
    model = BasicConvTestModel()

    config = get_quantization_config_without_range_init()
    config['compression'].update({
        "weights": {
            "mode": "asymmetric",
            "per_channel": True,
            "bits": 4
        },
        "activations": {
            "mode": "asymmetric",
            "bits": 4,
            "signed": True,
        },
    })
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qconfig = QuantizerConfig(bits=4,
                                         mode=QuantizationMode.ASYMMETRIC,
                                         signedness_to_force=None,
                                         per_channel=True,
                                         input_shape=None,
                                         is_weights=True)
    for wq in weight_quantizers.values():
        compare_qconfigs(ref_weight_qconfig, wq)

    ref_activation_qconfig = QuantizerConfig(bits=4,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             signedness_to_force=True,
                                             per_channel=False,
                                             input_shape=None,
                                             is_weights=False)

    for aq_info in activation_quantizer_infos.values():
        compare_qconfigs(ref_activation_qconfig, aq_info.quantizer_module_ref)


def compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name):
    def get_name(name):
        return '/'.join([model_name, name])

    all_quantizations = {str(key): quantizer for key, quantizer in algo.all_quantizations.items()}
    assert len(actual_pairs) == len(ref_pair_names)
    for (wqs, aq), (wqs_names, aq_name) in zip(actual_pairs, ref_pair_names):
        assert not aq.is_weights
        assert aq == all_quantizations[get_name(aq_name)]
        ref_weight_quantizers = [all_quantizations[get_name(name)] for name in wqs_names]
        for weight_quantizer in wqs:
            assert weight_quantizer.is_weights
            assert weight_quantizer in ref_weight_quantizers


#
#  fq           fq
#   \            \
# сonv0 - fq - conv1
#   /
# fq
#
def test_get_weight_activation_pairs():
    model_cls = TwoConvTestModel
    config = get_quantization_config_without_range_init()
    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['Sequential[features]/Sequential[0]/NNCFConv2d[0]module_weight'],
                       'Sequential[features]/Sequential[0]/NNCFConv2d[0]module_input',
                       ),
                      (['Sequential[features]/Sequential[1]/NNCFConv2d[0]module_weight'],
                       'Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0',
                       )]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_cls.__name__)


class DoubleWeightsPerActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.conv1 = create_conv(1, 2, 2, -1, -2)
        self.conv2 = create_conv(1, 2, 2, -1, -2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return self.conv1(x), self.conv2(x)


#              fq
#             /
#          conv2d
#         /
# relu - fq     fq
#         \    /
#         conv2d
#
def test_get_weight_activation_pairs__with_double_weights_per_activation():
    model_cls = DoubleWeightsPerActivation
    model_name = model_cls.__name__
    config = get_quantization_config_without_range_init()

    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['NNCFConv2d[conv1]module_weight', 'NNCFConv2d[conv2]module_weight'],
                       'ReLU[relu]/RELU_0')]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name)


class DoubleWeightsPerActivationWithExtraModule(DoubleWeightsPerActivation):
    def forward(self, x):
        x = self.relu(x)
        return self.conv1(torch.sigmoid(x)), self.conv2(torch.sigmoid(x))


#                     fq
#                      \
#         sigmoid - conv1d
#         /
# relu - fq           fq
#         \            \
#         sigmoid - conv2d
#
def test_get_weight_activation_pairs__with_extra_module():
    model_cls = DoubleWeightsPerActivationWithExtraModule
    model_name = model_cls.__name__
    config = get_quantization_config_without_range_init()
    config["compression"].update({
        "quantizable_subgraph_patterns": [["sigmoid", "conv2d"]],
        "quantize_inputs": False})

    _, algo = create_compressed_model_and_algo_for_test(model_cls(), config)

    actual_pairs = algo.get_weights_activation_quantizers_pairs()
    ref_pair_names = [(['NNCFConv2d[conv1]module_weight', 'NNCFConv2d[conv2]module_weight'],
                       'ReLU[relu]/RELU_0')]

    compare_weights_activation_quantizers_pairs(actual_pairs, algo, ref_pair_names, model_name)


def test_can_load_quant_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    compression_algo_builder_list = create_compression_algorithm_builders(config)
    assert len(compression_algo_builder_list) == 1
    assert isinstance(compression_algo_builder_list[0], QuantizationBuilder)

    quant_model, _ = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    quant_model_conv = get_all_modules_by_type(quant_model.get_nncf_wrapped_model(), 'NNCFConv2d')
    assert len(model_conv) == len(quant_model_conv)

    for module_scope, _ in model_conv.items():
        quant_scope = deepcopy(module_scope)  # type: Scope
        quant_scope.pop()
        quant_scope.push(ScopeElement('NNCFConv2d', 'conv'))
        assert quant_scope in quant_model_conv.keys()

        store = []
        for op in quant_model_conv[quant_scope].pre_ops.values():
            if isinstance(op, (UpdateInputs, UpdateWeight)) and isinstance(op.operand, SymmetricQuantizer):
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)
        assert UpdateWeight.__name__ in store


def test_can_create_quant_loss_and_scheduler():
    config = get_quantization_config_without_range_init()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)

    loss = compression_ctrl.loss
    assert isinstance(loss, CompressionLoss)

    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, CompressionScheduler)


def get_path_to_keys(tmp_path, rank):
    return '{}_{}'.format(tmp_path, str(rank))


def activation_quantizers_dumping_worker(current_gpu, config, tmp_path):
    model = resnet50(pretrained=False)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)
    path = get_path_to_keys(tmp_path, current_gpu)
    print(path)
    with open(path, 'w') as f:
        f.writelines("%s\n" % key for key in quant_model.activation_quantizers.keys())


def test_activation_quantizers_order_is_the_same__for_resnet50(tmp_path):
    config = get_empty_config(input_sample_sizes=[1, 3, 224, 224])
    config['compression'] = {'algorithm': 'quantization', "initializer": {"range": {"num_init_steps": 0}}}
    ngpus_per_node = torch.cuda.device_count()

    torch.multiprocessing.spawn(activation_quantizers_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(config, tmp_path),
                                join=True)

    with open(get_path_to_keys(tmp_path, 0), 'r') as f:
        ref_list = f.readlines()
    for i in range(1, ngpus_per_node):
        with open(get_path_to_keys(tmp_path, i), 'r') as f:
            curr_list = f.readlines()
            assert curr_list == ref_list


def test_load_state_sets_initialized_flag():
    config = get_quantization_config_without_range_init()

    model = TwoConvTestModel()
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    load_state(quant_model, {
        'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([1.0]),  # quantizer of 1st conv's weights
        'module.features.1.0.pre_ops.0.op.scale': torch.tensor([1.0])  # quantizer of 2nd conv's weights
    })

    quantizers = get_all_modules_by_type(quant_model, 'SymmetricQuantizer')
    for scope, module in quantizers.items():
        if 'activation_quantizers' in str(scope) or 'UpdateInputs' in str(scope):
            assert not module.initialized
        else:
            assert module.initialized


def test_quantize_has_proper_is_weights_flag():
    class Model(nn.Module):
        def __init__(self, size=1):
            super().__init__()
            self.size = size
            self.conv = nn.Conv2d(size, size, size)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    config = get_quantization_config_without_range_init(model_size=2)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    for module in quant_model.modules():
        if isinstance(module, NNCFConv2d):
            for op in module.pre_ops.values():
                assert isinstance(op, (UpdateWeight, UpdateInputs))
                assert op.operand.is_weights == isinstance(op, UpdateWeight)
    for _, aq in quant_model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER).items():
        assert aq.is_weights is False


def test_can_quantize_free_operators(mocker):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones([1]))
            self.bias = nn.Parameter(torch.ones([1]))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    mod = Model()
    config = get_quantization_config_without_range_init(model_size=1)

    config["compression"].update({"quantize_inputs": False})
    quant_model, _ = create_compressed_model_and_algo_for_test(mod, config)

    quantizer_list = quant_model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER).values()
    assert len(quantizer_list) == 2
    for quantizer in quantizer_list:
        mocker.spy(quantizer, 'quantize')

    quant_model.do_dummy_forward()
    for quantizer in quantizer_list:
        assert quantizer.quantize.call_count == 1


@pytest.fixture(name="hw_config_type", params=HWConfigType)
def hw_config_type_(request):
    return request.param


def test_hw_config_quantization_can_quantize_squeezenet(hw_config_type):
    config = get_squeezenet_quantization_config()
    config["hw_config"] = hw_config_type.value
    model = squeezenet1_1_custom()
    create_compressed_model_and_algo_for_test(model, config)


class QuantizeInputsTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=2)
        self.linear = nn.Linear(in_features=8, out_features=8)

    #    (1)     (2)      (3)    (4)   (5)
    #     |       |        |      |     |-----\
    #  (conv1)   (MP)     (MP)    (MP)  (MP)  |
    #     |       |        |      |     |     |
    #     |       |       (+)     |     |     |
    #     |       |--\     |      |     |     |
    #     |       |   \    |      |     |     |
    #     |    (conv2) | (conv3)  |     |     |
    #     |       |    |   |       \   /      |
    #     |     (AvP)  \   |       (cat)      |
    #     |       |     \  |         |        |
    #  (conv4) (linear)  \ |      (conv6)     |
    #     |       |      (cat)       |        |
    #     |       |        |        (+)------/
    #     |       |      (conv5)     |
    #   (AvP)     |        |         |
    #     |       |      (AvP)       |
    #      \      |        /         |
    #       \---(cat)---------------/

    def forward(self, input_1, input_2, input_3, input_4, input_5):
        x_1 = self.conv1(input_1)
        x_1 = self.conv4(x_1)
        x_1 = F.adaptive_avg_pool2d(x_1, output_size=1)
        x_1 = x_1.flatten(start_dim=1)

        x_2_br = F.max_pool2d(input_2, kernel_size=2)
        x_2 = self.conv2(x_2_br)
        x_2 = F.adaptive_avg_pool2d(x_2, output_size=1)
        x_2 = x_2.flatten(start_dim=1)
        x_2 = self.linear(x_2)

        x_3 = F.max_pool2d(input_3, kernel_size=2)
        x_3 = x_3 + torch.ones_like(x_3)
        x_3 = self.conv3(x_3)
        x_3 = x_3.flatten(start_dim=1)
        x_2_br = x_2_br.flatten(start_dim=1)
        x_3 = torch.cat([x_2_br, x_3], dim=-1)
        x_3 = self.conv5(x_3.unsqueeze(2).unsqueeze(3).transpose(1, 2))
        x_3 = F.adaptive_avg_pool2d(x_3, output_size=1)
        x_3 = x_3.flatten(start_dim=1)

        x_4 = F.max_pool2d(input_4, kernel_size=2)
        x_5 = F.max_pool2d(input_5, kernel_size=2)
        x_45 = torch.cat([x_4, x_5], dim=1)
        x_45 = self.conv6(x_45)
        x_45 = x_45.flatten(start_dim=1)
        in_5_flat = input_5.flatten(start_dim=1)
        x_45 += F.pad(input_5.flatten(start_dim=1), [0, x_45.shape[1] - in_5_flat.shape[1]])

        return torch.cat([x_1, x_2, x_3, x_45], dim=-1)


def test_quantize_inputs():
    model = QuantizeInputsTestModel()
    config = get_quantization_config_without_range_init()
    config["input_info"] = [
        {
            "sample_size": [2, 3, 32, 32],
        },
        {
            "sample_size": [2, 3, 32, 32],
        },
        {
            "sample_size": [2, 3, 32, 32],
        },
        {
            "sample_size": [2, 3, 32, 32],
        },
        {
            "sample_size": [2, 3, 32, 32],
        }
    ]

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_INPUT_MODULE_SCOPES = [
        "QuantizeInputsTestModel/NNCFConv2d[conv1]",
        "QuantizeInputsTestModel/NNCFConv2d[conv2]",
        "QuantizeInputsTestModel/NNCFConv2d[conv5]",
        "QuantizeInputsTestModel/NNCFConv2d[conv6]",
    ]
    for ref_qinput_module_scope_str in REF_QUANTIZED_INPUT_MODULE_SCOPES:
        scope = Scope.from_str(ref_qinput_module_scope_str)
        assert model.get_module_by_scope(scope) is not None
        assert ref_qinput_module_scope_str in compression_ctrl.quantized_inputs_modules_registry

    nncf_modules_dict = model.get_nncf_modules()
    for scope, nncf_module in nncf_modules_dict.items():
        scope_str = str(scope)
        update_inputs_count = sum(1 for pre_op in nncf_module.pre_ops.values() if isinstance(pre_op, UpdateInputs))
        if scope_str in REF_QUANTIZED_INPUT_MODULE_SCOPES:
            assert update_inputs_count == 1
        else:
            assert update_inputs_count == 0

def make_op_exec_context_for_coalescing_test(scope_str: str) -> OperationExecutionContext:
    ia_op_exec_context = InputAgnosticOperationExecutionContext.from_str(scope_str)
    op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                ia_op_exec_context.scope_in_model,
                                                ia_op_exec_context.call_order,
                                                [TensorMeta(0, 0, [1])])
    return op_exec_context

def make_insertion_info_for_coalescing_test(scope_str: str,
                                            linked_op_exec_contexts: List[OperationExecutionContext] = None):
    op_exec_context = make_op_exec_context_for_coalescing_test(scope_str)
    retval = InsertionInfo(op_exec_context)
    if linked_op_exec_contexts is not None:
        retval.linked_op_exec_contexts = linked_op_exec_contexts
    return retval


@pytest.mark.parametrize("input_insertion_infos, linked_scopes_groups_list, ref_coalesced_insertion_infos",
                         # ref_coalesced_insertion_infos == None means that the coalescing should raise an exception
                         [
                             # 0 - Empty linked scopes list
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [],
                                 # Same as input
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                             ),
                             # 1 - Linked scope only affects 1 operation
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"]],
                                 # Same as input
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ]
                             ),
                             # 2 - Same as 1 but with multiple groups
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"], ["Foo/Xyz[leet]/__add___0"]],
                                 # Same as input again
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ]
                             ),
                             # 3 - Single group affecting some of the scopes
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     )
                                 ],
                                 [["Foo/Xyz[leet]/matmul_0", "Foo/Xyz[leet]/__add___0", "Foo/Baz[bar]/linear_0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/__add___0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     )
                                 ]
                             ),

                             # 4 - Multiple groups, each affecting one operation
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/linear_0"], ["Foo/Asdf[jkl]/softmax_0"]],
                                 [
                                     # Same as input
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ]
                             ),

                             # 5 - Multiple groups affecting multiple operations without overlapping
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_1"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0",
                                   "Foo/Baz[bar]/linear_0"],
                                  ["Foo/Asdf[jkl]/softmax_1", "Foo/Xyz[leet]/__add___0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_1",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/__add___0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ]
                             ),

                             # 6 - A variation of 5
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Asdf[jkl]/softmax_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/matmul_0"
                                             )
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                 ]
                             ),

                             # 7 - Overlapping groups
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),

                             # 8 - More than 1 match for the operation specified in the group

                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),

                             # 9 - No match for an operation specified in the group
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_1"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),
                         ])
def test_insertion_info_coalescing(input_insertion_infos: List[InsertionInfo],
                                   linked_scopes_groups_list: List[List[str]],
                                   ref_coalesced_insertion_infos: List[InsertionInfo]):
    if ref_coalesced_insertion_infos is None:
        with pytest.raises(RuntimeError):
            _ = QuantizationBuilder.coalesce_insertion_infos(input_insertion_infos,
                                                             linked_scopes_groups_list)
    else:
        test_coalesced_insertion_infos = QuantizationBuilder.coalesce_insertion_infos(input_insertion_infos,
                                                                                      linked_scopes_groups_list)
        assert Counter(test_coalesced_insertion_infos) == Counter(ref_coalesced_insertion_infos)


class QuantizerLinkingTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_trainable_param = torch.nn.Parameter(torch.ones([1]))

        class Path(torch.nn.Module):
            def forward(self, input_1, input_2):
                retval0 = input_1 + input_2
                retval1 = retval0 * input_2
                retval2 = retval0 + retval1
                # __add___0, __mul___0, __add___1 results respectively
                return retval0, retval1, retval2

        self.path1 = Path()
        self.path2 = Path()

    def forward(self, input_1, input_2):
        path1_results = self.path1(input_1, input_2)
        path2_results = self.path2(input_1, input_2)
        return tuple(x + y for x, y in zip(path1_results, path2_results))


def test_quantizer_scale_linking():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["compression"]["quantize_outputs"] = True
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 1, 1, 1],
        },
        {
            "sample_size": [1, 1, 1, 1],
        }
    ]
    nncf_config["compression"]["activations"] = {
        "linked_quantizer_scopes": [
            [
                # Note: Assuming that quantizers are attached as a post-op to the specified operation
                "QuantizerLinkingTestModel/Path[path2]/__mul___0",
                "QuantizerLinkingTestModel/Path[path2]/__add___0",
            ]
        ],
        "ignored_scopes": [
            # Ignore path output averaging operations
            "QuantizerLinkingTestModel/__add___0",
            "QuantizerLinkingTestModel/__add___1",
            "QuantizerLinkingTestModel/__add___2",
        ]
    }

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(QuantizerLinkingTestModel(),
                                                                                   nncf_config)

    # 2 paths x 3 quantizers - 1 because two are shared in one path
    assert len(compression_ctrl.non_weight_quantizers) == 5

    test_input1 = torch.ones([1, 1, 1, 1])
    test_input2 = 2 * test_input1

    non_shared_mul_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path1]/__mul___0"))

    non_shared_add_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path1]/__add___0"))

    shared_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path2]/__mul___0"))

    non_shared_mul_quantizer = compression_ctrl.non_weight_quantizers[non_shared_mul_quantizer_id].quantizer_module_ref
    non_shared_add_quantizer = compression_ctrl.non_weight_quantizers[non_shared_add_quantizer_id].quantizer_module_ref
    shared_quantizer = compression_ctrl.non_weight_quantizers[shared_quantizer_id].quantizer_module_ref

    old_scale = 765.0  # so that the quantum is equal to 3
    with torch.no_grad():
        for quantizer in compression_ctrl.all_quantizations.values():
            quantizer.scale.fill_(old_scale)


    # Expected outputs without compression - 6, 12, 8. Scale deliberately set to preserve the values
    uncompressed_expected_outputs = (6.0 * torch.ones([1]), 12.0 * torch.ones([1]), 18.0 * torch.ones([1]))
    outputs_with_shared_scale_1 = compressed_model(test_input1, test_input2)

    for uncomp_out, comp_out_1 in zip(uncompressed_expected_outputs, outputs_with_shared_scale_1):
        assert torch.allclose(uncomp_out, comp_out_1)

    # Specifically clip the shared quantizer's outputs by setting scale to 1.0
    new_shared_scale = 1.0
    with torch.no_grad():
        shared_quantizer.scale.fill_(new_shared_scale)
    outputs_with_shared_scale_2 = compressed_model(test_input1, test_input2)

    # __add___0 outputs
    assert torch.allclose(outputs_with_shared_scale_2[0], 4.0 * torch.ones([1]))
    # __mul___0 outputs
    assert torch.allclose(outputs_with_shared_scale_2[1], 7.0 * torch.ones([1]))
    # __add___1 outputs
    assert torch.allclose(outputs_with_shared_scale_2[2], 12.0 * torch.ones([1]))

    # Clipping the non-shared quantizers at the same position in the path as the two shared ones
    # in the same manner is required to simulate the same grad input for both the shared quantizers
    # and the unshared ones
    with torch.no_grad():
        non_shared_mul_quantizer.scale.fill_(new_shared_scale)
        non_shared_add_quantizer.scale.fill_(new_shared_scale)
    final_output = compressed_model(test_input1, test_input2)[2]
    final_output.backward()

    assert torch.allclose(shared_quantizer.scale.grad,
                          non_shared_mul_quantizer.scale.grad + non_shared_add_quantizer.scale.grad)
