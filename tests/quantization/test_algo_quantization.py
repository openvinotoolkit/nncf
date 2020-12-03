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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from copy import deepcopy

from torchvision.models import resnet50, squeezenet1_1

from nncf.checkpoint_loading import load_state
from nncf.compression_method_api import CompressionLoss, CompressionScheduler
from nncf.dynamic_graph.context import ScopeElement, Scope
from nncf.hw_config import HWConfigType
from nncf.layers import NNCFConv2d
from nncf.model_creation import create_compression_algorithm_builders
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_network import ExtraCompressionModuleType
from nncf.quantization.algo import QuantizationController, QuantizationBuilder
from nncf.quantization.layers import QuantizationMode, QuantizerConfig, SymmetricQuantizer, BaseQuantizer, \
    QUANTIZATION_MODULES
from nncf.utils import get_all_modules_by_type
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init, \
    get_squeezenet_quantization_config
from tests.helpers import BasicConvTestModel, TwoConvTestModel, get_empty_config, \
    create_compressed_model_and_algo_for_test, create_conv


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

    ref_weight_qconfig = QuantizerConfig(8, QuantizationMode.SYMMETRIC, True, True, None, True)
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
    config['target_device'] = 'TRIAL'
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
    def get_wq_name(name):
        return '/'.join([model_name, name])

    def get_aq_name(name):
        if name == '/nncf_model_input_0':
            return name  + '|OUTPUT'
        return '/'.join([model_name, name]) + '|OUTPUT'

    all_quantizations = {str(key): quantizer for key, quantizer in algo.all_quantizations.items()}
    assert len(actual_pairs) == len(ref_pair_names)
    for (wqs, aq), (wqs_names, aq_name) in zip(actual_pairs, ref_pair_names):
        assert not aq.is_weights
        assert aq == all_quantizations[get_aq_name(aq_name)]
        ref_weight_quantizers = [all_quantizations[get_wq_name(name)] for name in wqs_names]
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
                       '/nncf_model_input_0',
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
                       '/nncf_model_input_0')]

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
    config['quantizer_setup_type'] = 'pattern_based'
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
    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)

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
    config['compression'] = {'algorithm': 'quantization', "initializer": {"range": {"num_init_samples": 0}}}
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
    for _, aq in quant_model.get_compression_modules_by_type(ExtraCompressionModuleType.ACTIVATION_QUANTIZER).items():
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

    quantizer_list = quant_model.get_compression_modules_by_type(ExtraCompressionModuleType.FUNCTION_QUANTIZER).values()
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
    model = squeezenet1_1()
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

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_INPUT_MODULE_SCOPES = [
        '/nncf_model_input_0',
        '/nncf_model_input_1',
        '/nncf_model_input_2',
        '/nncf_model_input_3',
        '/nncf_model_input_4'
    ]
    actual_input_quantizer_str_scopes =\
         [str_scope for str_scope in model.activation_quantizers if 'nncf_model_input' in str_scope]
    assert len(REF_QUANTIZED_INPUT_MODULE_SCOPES) == len(actual_input_quantizer_str_scopes)
    for ref_qinput_scope_str in REF_QUANTIZED_INPUT_MODULE_SCOPES:
        assert isinstance(model.activation_quantizers[ref_qinput_scope_str], SymmetricQuantizer)
