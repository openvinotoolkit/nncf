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
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import List
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.models import resnet50
from torchvision.models import squeezenet1_1

from nncf.api.compression import CompressionScheduler
from nncf.torch.checkpoint_loading import load_state
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.common.hardware.config import HWConfigType
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdateInputs
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import ExtraCompressionModuleType
from nncf.torch.quantization.algo import QuantizationBuilder
from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.torch.utils import get_all_modules_by_type
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.test_quantization_helpers import get_squeezenet_quantization_config


def compare_qspecs(qspec: PTQuantizerSpec, quantizer: BaseQuantizer):
    assert qspec.narrow_range == quantizer.narrow_range
    assert qspec.num_bits == quantizer.num_bits
    assert isinstance(quantizer, QUANTIZATION_MODULES.get(qspec.mode))
    assert qspec.scale_shape == quantizer.scale_shape
    #pylint:disable=protected-access
    assert qspec.signedness_to_force == quantizer._signedness_to_force


def test_quantization_configs__with_defaults():
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qspec = PTQuantizerSpec(num_bits=8,
                                       mode=QuantizationMode.SYMMETRIC,
                                       signedness_to_force=True,
                                       narrow_range=True,
                                       half_range=False,
                                       scale_shape=model.wq_scale_shape_per_channel,
                                       logarithm_scale=False)
    for wq_info in weight_quantizers.values():
        compare_qspecs(ref_weight_qspec, wq_info.quantizer_module_ref)

    ref_activation_qspec = PTQuantizerSpec(num_bits=8,
                                           mode=QuantizationMode.SYMMETRIC,
                                           signedness_to_force=None,
                                           narrow_range=False,
                                           half_range=False,
                                           scale_shape=(1, ),
                                           logarithm_scale=False)
    for aq_info in activation_quantizer_infos.values():
        compare_qspecs(ref_activation_qspec, aq_info.quantizer_module_ref)


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
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qspec = PTQuantizerSpec(num_bits=4,
                                       mode=QuantizationMode.ASYMMETRIC,
                                       signedness_to_force=None,
                                       scale_shape=model.wq_scale_shape_per_channel,
                                       narrow_range=True,
                                       half_range=False,
                                       logarithm_scale=False)
    for wq_info in weight_quantizers.values():
        compare_qspecs(ref_weight_qspec, wq_info.quantizer_module_ref)

    ref_activation_qspec = PTQuantizerSpec(num_bits=4,
                                           mode=QuantizationMode.ASYMMETRIC,
                                           signedness_to_force=True,
                                           scale_shape=(1, ),
                                           narrow_range=False,
                                           half_range=False,
                                           logarithm_scale=False)

    for aq_info in activation_quantizer_infos.values():
        compare_qspecs(ref_activation_qspec, aq_info.quantizer_module_ref)


def compare_weights_activation_quantizers_pairs(actual_pairs: List[Tuple[List[WeightQuantizerId],
                                                                         NonWeightQuantizerId]],
                                                algo, ref_pair_names, model_name):
    def get_wq_name(name):
        return '/'.join([model_name, name])

    def get_aq_name(name):
        if name == '/nncf_model_input_0':
            return name  + '|OUTPUT'
        return '/'.join([model_name, name]) + '|OUTPUT'

    all_quantizations = {str(key): quantizer for key, quantizer in algo.all_quantizations.items()}
    assert len(actual_pairs) == len(ref_pair_names)
    for (wq_ids, aq_id), (wqs_names, aq_name) in zip(actual_pairs, ref_pair_names):
        wqs = [algo.all_quantizations[wq_id] for wq_id in wq_ids]
        aq = algo.all_quantizations[aq_id]
        assert not aq.narrow_range
        assert aq == all_quantizations[get_aq_name(aq_name)]
        ref_weight_quantizers = [all_quantizations[get_wq_name(name)] for name in wqs_names]
        for weight_quantizer in wqs:
            assert weight_quantizer.narrow_range
            assert weight_quantizer in ref_weight_quantizers


def test_can_load_quant_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    register_bn_adaptation_init_args(config)
    composite_builder = PTCompositeCompressionAlgorithmBuilder(config)
    assert len(composite_builder.child_builders) == 1
    assert isinstance(composite_builder.child_builders[0], QuantizationBuilder)

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
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)

    loss = compression_ctrl.loss
    assert isinstance(loss, PTCompressionLoss)

    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, CompressionScheduler)


def get_path_to_keys(tmp_path, rank):
    return '{}_{}'.format(tmp_path, str(rank))


def activation_quantizers_dumping_worker(current_gpu, config, tmp_path):
    model = resnet50(pretrained=False)
    _, qctrl = create_compressed_model_and_algo_for_test(model, config)
    path = get_path_to_keys(tmp_path, current_gpu)
    print(path)
    with open(path, 'w') as f:
        for aq_id in qctrl.non_weight_quantizers:
            f.writelines("%s\n" % str(aq_id))


def test_activation_quantizers_order_is_the_same__for_resnet50(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    config = get_empty_config(input_sample_sizes=[1, 3, 224, 224])
    config['compression'] = {'algorithm': 'quantization', "initializer": {"range": {"num_init_samples": 0}}}
    register_bn_adaptation_init_args(config)
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
    register_bn_adaptation_init_args(config)

    model = TwoConvTestModel()
    quant_model, qctrl = create_compressed_model_and_algo_for_test(model, config)

    load_state(quant_model, {
        'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([1.0]),  # quantizer of 1st conv's weights
        'module.features.1.0.pre_ops.0.op.scale': torch.ones(1, 1, 1, 1)  # quantizer of 2nd conv's weights
    })

    for wq_info in qctrl.weight_quantizers.values():
        assert wq_info.quantizer_module_ref.initialized

    for aq_info in qctrl.non_weight_quantizers.values():
        assert not aq_info.quantizer_module_ref.initialized


def test_quantizers_have_proper_narrow_range_set():
    class Model(nn.Module):
        def __init__(self, size=1):
            super().__init__()
            self.size = size
            self.conv = nn.Conv2d(size, size, size)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    config = get_quantization_config_without_range_init(model_size=2)
    register_bn_adaptation_init_args(config)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    for module in quant_model.modules():
        if isinstance(module, NNCFConv2d):
            for op in module.pre_ops.values():
                assert isinstance(op, (UpdateWeight, UpdateInputs))
                assert op.operand.narrow_range == isinstance(op, UpdateWeight)
    for _, aq in quant_model.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER).items():
        assert aq.narrow_range is False


@pytest.fixture(name="hw_config_type", params=HWConfigType)
def hw_config_type_(request):
    return request.param


def test_hw_config_quantization_can_quantize_squeezenet(hw_config_type):
    config = get_squeezenet_quantization_config()
    config["target_device"] = hw_config_type.value
    register_bn_adaptation_init_args(config)
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
    register_bn_adaptation_init_args(config)

    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_INPUT_MODULE_SCOPES = [
        '/nncf_model_input_0|OUTPUT',
        '/nncf_model_input_1|OUTPUT',
        '/nncf_model_input_2|OUTPUT',
        '/nncf_model_input_3|OUTPUT',
        '/nncf_model_input_4|OUTPUT'
    ]

    actual_input_quantizer_str_scopes = []
    for aq_id, aq_info in qctrl.non_weight_quantizers.items():
        for target_point in aq_info.affected_insertions:
            quantizer_target_node_name = str(target_point.target_node_name)
            if 'nncf_model_input' in quantizer_target_node_name:
                actual_input_quantizer_str_scopes.append(quantizer_target_node_name)

    assert len(REF_QUANTIZED_INPUT_MODULE_SCOPES) == len(actual_input_quantizer_str_scopes)
    for qinput_scope_str in actual_input_quantizer_str_scopes:
        matches = set()
        for aq_id, aq_info in qctrl.non_weight_quantizers.items():
            for target_point in aq_info.affected_insertions:
                if qinput_scope_str in str(target_point.target_node_name):
                    matches.add(aq_id)
        assert len(matches) == 1
        input_aq_id = next(iter(matches))
        quantizer = qctrl.non_weight_quantizers[input_aq_id].quantizer_module_ref
        assert isinstance(quantizer, SymmetricQuantizer)



@pytest.mark.parametrize(
    ('requanting_qconf', 'base_qconf', 'is_valid_requant'),
    (
        (QuantizerConfig(), QuantizerConfig(), True),

        (QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6), False),
        (QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8), True),

        # Technically placing a per-channel quantization after a per-tensor should not break
        # anything or limit the set of output values w.r.t to a single per-tensor quantizer.
        (QuantizerConfig(num_bits=6, per_channel=True), QuantizerConfig(num_bits=6, per_channel=False), True),
        (QuantizerConfig(num_bits=6, per_channel=False), QuantizerConfig(num_bits=6, per_channel=True), True),

        (QuantizerConfig(num_bits=5, per_channel=True), QuantizerConfig(num_bits=6, per_channel=False), True),
        (QuantizerConfig(num_bits=5, per_channel=False), QuantizerConfig(num_bits=6, per_channel=True), True),

        (
                QuantizerConfig(num_bits=5, mode=QuantizationMode.SYMMETRIC),
                QuantizerConfig(num_bits=5, mode=QuantizationMode.ASYMMETRIC),
                True
        ),
        (
                QuantizerConfig(num_bits=5, mode=QuantizationMode.ASYMMETRIC),
                QuantizerConfig(num_bits=5, mode=QuantizationMode.SYMMETRIC),
                False
        ),


        (QuantizerConfig(signedness_to_force=True), QuantizerConfig(), True),
        (QuantizerConfig(), QuantizerConfig(signedness_to_force=True), False),

        (QuantizerConfig(signedness_to_force=False), QuantizerConfig(), True),
        (QuantizerConfig(), QuantizerConfig(signedness_to_force=False), False),

        (QuantizerConfig(signedness_to_force=True), QuantizerConfig(signedness_to_force=False), False),
        (QuantizerConfig(signedness_to_force=False), QuantizerConfig(signedness_to_force=True), True),

        (
            QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
            True
        ),

        (
            QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
            True
        ),

        # Neither of the two configs here can requantize the other
        (
            QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
            QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
            False
        ),
        (
            QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
            QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
            False
        )
    )
)
def test_quantizer_ordering(requanting_qconf: QuantizerConfig,
                            base_qconf: QuantizerConfig, is_valid_requant: bool):
    test_result = requanting_qconf.is_valid_requantization_for(base_qconf)
    assert test_result == is_valid_requant

class QuantizeOutputsTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)


    def forward(self, x):
        self.conv5(x)
        return self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)


def test_quantize_outputs():
    config = get_quantization_config_without_range_init()
    config["input_info"] = [
        {
            "sample_size": [2, 3, 32, 32],
        }
    ]
    model = QuantizeOutputsTestModel()
    config['compression']['quantize_outputs'] = True
    register_bn_adaptation_init_args(config)
    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_OUTPUT_MODULE_SCOPES = [
        'QuantizeOutputsTestModel/NNCFConv2d[conv1]/conv2d_0|OUTPUT',
        'QuantizeOutputsTestModel/NNCFConv2d[conv2]/conv2d_0|OUTPUT',
        'QuantizeOutputsTestModel/NNCFConv2d[conv3]/conv2d_0|OUTPUT',
        'QuantizeOutputsTestModel/NNCFConv2d[conv4]/conv2d_0|OUTPUT'
    ]
    actual_output_quantizer_str_scopes =\
         [str(aq_id) for aq_id in qctrl.non_weight_quantizers if 'nncf_model_input' not in str(aq_id)]
    assert len(REF_QUANTIZED_OUTPUT_MODULE_SCOPES) == len(actual_output_quantizer_str_scopes)

    for ref_qinput_scope_str in REF_QUANTIZED_OUTPUT_MODULE_SCOPES:
        matches = []
        for aq_id in qctrl.non_weight_quantizers:
            if str(aq_id) == ref_qinput_scope_str:
                matches.append(aq_id)
        assert len(matches) == 1
        quantizer = qctrl.non_weight_quantizers[matches[0]].quantizer_module_ref
        assert isinstance(quantizer, SymmetricQuantizer)


def test_quantize_outputs_with_scope_overrides():
    config = get_quantization_config_without_range_init()
    config["input_info"] = [
        {
            "sample_size": [2, 3, 32, 32],
        }
    ]
    model = QuantizeOutputsTestModel()
    config['compression']['quantize_outputs'] = True
    config['target_device'] = "TRIAL"
    config['compression']['scope_overrides'] = {
        "activations": {
            "/nncf_model_output_0": {
                "bits": 4,
                "mode": "asymmetric",
            }
        }
    }
    register_bn_adaptation_init_args(config)
    model, ctrl = create_compressed_model_and_algo_for_test(model, config)
    output_quantizers =\
        [q for qid, q in ctrl.all_quantizations.items() if isinstance(qid, NonWeightQuantizerId)][:-1]
    for q in output_quantizers:
        assert q.num_bits == 4
        assert isinstance(q, AsymmetricQuantizer)


@contextmanager
def nncf_debug():
    from nncf.torch import set_log_level
    set_log_level(logging.DEBUG)
    yield
    set_log_level(logging.INFO)


def test_debug_mode():
    config = get_quantization_config_without_range_init()
    register_bn_adaptation_init_args(config)
    model = BasicConvTestModel()
    with nncf_debug():
        model, _ = create_compressed_model_and_algo_for_test(model, config)
        model.forward(torch.zeros(BasicConvTestModel.INPUT_SIZE,
                                  device=next(model.parameters()).device))
