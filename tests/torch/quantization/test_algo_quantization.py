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
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import autocast
from torch import nn
from torchvision.models import resnet50
from torchvision.models import squeezenet1_1

from nncf import NNCFConfig
from nncf.api.compression import CompressionScheduler
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.utils.debug import nncf_debug
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args
from nncf.torch import register_module
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.layers import NNCFConv2d
from nncf.torch.model_creation import create_compression_algorithm_builder
from nncf.torch.module_operations import UpdateInputs
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.quantization.algo import QuantizationBuilder
from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.utils import get_model_device
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import LeNet
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import create_random_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.quantization_helpers import get_squeezenet_quantization_config


def compare_qspecs(qspec: PTQuantizerSpec, quantizer: BaseQuantizer):
    assert qspec.narrow_range == quantizer.narrow_range
    assert qspec.num_bits == quantizer.num_bits
    assert isinstance(quantizer, QUANTIZATION_MODULES.get(qspec.mode))
    assert qspec.scale_shape == quantizer.scale_shape

    assert qspec.signedness_to_force == quantizer._signedness_to_force


def test_quantization_configs__with_defaults():
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init()
    config["compression"]["overflow_fix"] = "disable"
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=True,
        half_range=False,
        scale_shape=model.wq_scale_shape_per_channel,
        logarithm_scale=False,
    )
    for wq_info in weight_quantizers.values():
        compare_qspecs(ref_weight_qspec, wq_info.quantizer_module_ref)

    ref_activation_qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        narrow_range=False,
        half_range=False,
        scale_shape=(1,),
        logarithm_scale=False,
    )
    for aq_info in activation_quantizer_infos.values():
        compare_qspecs(ref_activation_qspec, aq_info.quantizer_module_ref)


def test_quantization_configs__custom():
    model = BasicConvTestModel()

    config = get_quantization_config_without_range_init()
    config["compression"].update(
        {
            "weights": {"mode": "asymmetric", "per_channel": True, "bits": 4},
            "activations": {
                "mode": "asymmetric",
                "bits": 4,
                "signed": True,
            },
        }
    )
    config["target_device"] = "TRIAL"
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, QuantizationController)
    weight_quantizers = compression_ctrl.weight_quantizers
    activation_quantizer_infos = compression_ctrl.non_weight_quantizers

    ref_weight_qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        scale_shape=model.wq_scale_shape_per_channel,
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )
    for wq_info in weight_quantizers.values():
        compare_qspecs(ref_weight_qspec, wq_info.quantizer_module_ref)

    ref_activation_qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=True,
        scale_shape=(1,),
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )

    for aq_info in activation_quantizer_infos.values():
        compare_qspecs(ref_activation_qspec, aq_info.quantizer_module_ref)


def compare_weights_activation_quantizers_pairs(
    actual_pairs: List[Tuple[List[WeightQuantizerId], NonWeightQuantizerId]], algo, ref_pair_names, model_name
):
    def get_wq_name(name):
        return "/".join([model_name, name])

    def get_aq_name(name):
        if name == "/nncf_model_input_0":
            return name + "|OUTPUT"
        return "/".join([model_name, name]) + "|OUTPUT"

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
    builder = create_compression_algorithm_builder(config)
    assert isinstance(builder, QuantizationBuilder)

    quant_model, _ = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    model_conv = get_all_modules_by_type(model, "Conv2d")
    quant_model_conv = get_all_modules_by_type(quant_model, "NNCFConv2d")
    assert len(model_conv) == len(quant_model_conv)

    for module_scope, _ in model_conv.items():
        quant_scope: Scope = deepcopy(module_scope)
        quant_scope.pop()
        quant_scope.push(ScopeElement("NNCFConv2d", "conv"))
        assert quant_scope in quant_model_conv

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
    return "{}_{}".format(tmp_path, str(rank))


def activation_quantizers_dumping_worker(current_gpu, config, tmp_path):
    model = resnet50(pretrained=False)
    _, qctrl = create_compressed_model_and_algo_for_test(model, config)
    path = get_path_to_keys(tmp_path, current_gpu)
    print(path)
    with open(path, "w", encoding="utf8") as f:
        for aq_id in qctrl.non_weight_quantizers:
            f.writelines("%s\n" % str(aq_id))


@pytest.mark.cuda
def test_activation_quantizers_order_is_the_same__for_resnet50(tmp_path, runs_subprocess_in_precommit):
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    config = get_empty_config(input_sample_sizes=[1, 3, 224, 224])
    config["compression"] = {"algorithm": "quantization", "initializer": {"range": {"num_init_samples": 0}}}
    register_bn_adaptation_init_args(config)
    ngpus_per_node = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        activation_quantizers_dumping_worker, nprocs=ngpus_per_node, args=(config, tmp_path), join=True
    )

    with open(get_path_to_keys(tmp_path, 0), "r", encoding="utf8") as f:
        ref_list = f.readlines()
    for i in range(1, ngpus_per_node):
        with open(get_path_to_keys(tmp_path, i), "r", encoding="utf8") as f:
            curr_list = f.readlines()
            assert curr_list == ref_list


def test_load_state_sets_initialized_flag():
    config = get_quantization_config_without_range_init()
    register_bn_adaptation_init_args(config)

    model = TwoConvTestModel()
    quant_model, qctrl = create_compressed_model_and_algo_for_test(model, config)

    load_state(
        quant_model,
        {
            "module.features.0.0.pre_ops.0.op.signed_tensor": torch.tensor([1.0]),  # quantizer of 1st conv's weights
            "module.features.1.0.pre_ops.0.op.scale": torch.ones(1, 1, 1, 1),  # quantizer of 2nd conv's weights
        },
    )

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
    config["compression"]["overflow_fix"] = "disable"
    register_bn_adaptation_init_args(config)
    quant_model, _ = create_compressed_model_and_algo_for_test(model, config)

    for module in quant_model.modules():
        if isinstance(module, NNCFConv2d):
            for op in module.pre_ops.values():
                assert isinstance(op, (UpdateWeight, UpdateInputs))
                assert op.operand.narrow_range == isinstance(op, UpdateWeight)
    for _, aq in quant_model.nncf.get_compression_modules_by_type(
        ExtraCompressionModuleType.EXTERNAL_QUANTIZER
    ).items():
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
        },
    ]
    register_bn_adaptation_init_args(config)

    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    REF_QUANTIZED_INPUT_MODULE_SCOPES = [
        "/nncf_model_input_0|OUTPUT",
        "/nncf_model_input_1|OUTPUT",
        "/nncf_model_input_2|OUTPUT",
        "/nncf_model_input_3|OUTPUT",
        "/nncf_model_input_4|OUTPUT",
    ]

    actual_input_quantizer_str_scopes = []
    for aq_id, aq_info in qctrl.non_weight_quantizers.items():
        for target_point in aq_info.affected_insertions:
            quantizer_target_node_name = str(target_point.target_node_name)
            if "nncf_model_input" in quantizer_target_node_name:
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
    ("requanting_qconf", "base_qconf", "is_valid_requant"),
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
            True,
        ),
        (
            QuantizerConfig(num_bits=5, mode=QuantizationMode.ASYMMETRIC),
            QuantizerConfig(num_bits=5, mode=QuantizationMode.SYMMETRIC),
            False,
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
            True,
        ),
        (
            QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
            True,
        ),
        # Neither of the two configs here can requantize the other
        (
            QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
            QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
            False,
        ),
        (
            QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
            QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
            False,
        ),
    ),
)
def test_quantizer_ordering(requanting_qconf: QuantizerConfig, base_qconf: QuantizerConfig, is_valid_requant: bool):
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
    config["compression"]["quantize_outputs"] = True
    register_bn_adaptation_init_args(config)
    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    # The quantizers below will not have been set up due to quantizer propagation,
    # and no configuration can be determined for them from the HW config. The
    # configuration is also missing in this case in the NNCFConfig, so will
    # set up a quantizer with default config.
    REF_QUANTIZED_OUTPUT_MODULE_SCOPES = [
        "QuantizeOutputsTestModel/NNCFConv2d[conv1]/conv2d_0|OUTPUT",
        "QuantizeOutputsTestModel/NNCFConv2d[conv2]/conv2d_0|OUTPUT",
        "QuantizeOutputsTestModel/NNCFConv2d[conv3]/conv2d_0|OUTPUT",
        "QuantizeOutputsTestModel/NNCFConv2d[conv4]/conv2d_0|OUTPUT",
    ]
    actual_output_quantizer_str_scopes = [
        str(aq_id) for aq_id in qctrl.non_weight_quantizers if "nncf_model_input" not in str(aq_id)
    ]
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
    config["compression"]["quantize_outputs"] = True
    config["target_device"] = "TRIAL"
    config["compression"]["scope_overrides"] = {
        "activations": {
            "/nncf_model_output_0": {
                "bits": 4,
                "mode": "asymmetric",
            }
        }
    }
    register_bn_adaptation_init_args(config)
    model, ctrl = create_compressed_model_and_algo_for_test(model, config)
    output_quantizers = [q for qid, q in ctrl.all_quantizations.items() if isinstance(qid, NonWeightQuantizerId)]
    for q in output_quantizers[1:]:
        assert q.num_bits == 8
        assert isinstance(q, SymmetricQuantizer)

    assert output_quantizers[0].num_bits == 4
    assert isinstance(output_quantizers[0], AsymmetricQuantizer)


class IntermediateOutputModel(nn.Module):
    """
    When quantized with "quantize_outputs": False (which is the default behaviour),
    the activation quantizer of `conv2` shall not propagate to the output of `conv1`,
    but shall stay as a pre-hook to the `conv2`, so as not to impact the
    return value of `conv1` which is also an intermediate output of the model.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1, self.conv2(x1)


def test_intermediate_output_model():
    config = get_quantization_config_without_range_init()
    config["input_info"] = [
        {
            "sample_size": [2, 3, 32, 32],
        }
    ]
    model = IntermediateOutputModel()
    config["compression"]["quantize_outputs"] = False
    register_bn_adaptation_init_args(config)
    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    activation_quantizer_scopes = [str(aq_id) for aq_id in qctrl.non_weight_quantizers]
    assert Counter(activation_quantizer_scopes) == Counter(
        [
            "/nncf_model_input_0|OUTPUT",  # activation quantizer of conv1
            "IntermediateOutputModel/NNCFConv2d[conv2]/conv2d_0|INPUT0",
        ]
    )  # act. quant. of conv2


def test_debug_mode():
    config = get_quantization_config_without_range_init()
    register_bn_adaptation_init_args(config)
    model = BasicConvTestModel()
    with nncf_debug():
        model, _ = create_compressed_model_and_algo_for_test(model, config)
        model.forward(torch.zeros(BasicConvTestModel.INPUT_SIZE, device=get_model_device(model)))


class SharedLayersModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.shared_conv(x)
        x = x + x
        x = self.shared_conv(x)
        x = x * x
        return x


def test_shared_layers_are_weight_quantized_only_once():
    model = SharedLayersModel()
    config = get_quantization_config_without_range_init(model_size=1)
    register_bn_adaptation_init_args(config)
    model, qctrl = create_compressed_model_and_algo_for_test(model, config)
    assert len(qctrl.weight_quantizers) == 1


TEST_QUANTIZATION_PRESET_STRUCT = [
    {
        "preset": "performance",
        "target_device": "CPU",
        "overrided_param": {},
        "expected_weights_q": SymmetricQuantizer,
        "expected_activations_q": SymmetricQuantizer,
    },
    {
        "preset": "mixed",
        "target_device": "CPU",
        "overrided_param": {},
        "expected_weights_q": SymmetricQuantizer,
        "expected_activations_q": AsymmetricQuantizer,
    },
    {
        "preset": "performance",
        "target_device": "GPU",
        "overrided_param": {},
        "expected_weights_q": SymmetricQuantizer,
        "expected_activations_q": SymmetricQuantizer,
    },
    {
        "preset": "mixed",
        "target_device": "GPU",
        "overrided_param": {},
        "expected_weights_q": SymmetricQuantizer,
        "expected_activations_q": AsymmetricQuantizer,
    },
    {
        "preset": "performance",
        "target_device": "CPU",
        "overrided_param": {"weights": {"mode": "asymmetric"}},
        "expected_weights_q": AsymmetricQuantizer,
        "expected_activations_q": SymmetricQuantizer,
    },
]


@pytest.mark.parametrize("data", TEST_QUANTIZATION_PRESET_STRUCT)
def test_quantization_preset(data):
    model = BasicConvTestModel()
    config = get_empty_config(input_sample_sizes=[1, 1, 4, 4])
    config["target_device"] = data["target_device"]
    config["compression"] = {"algorithm": "quantization", "preset": data["preset"]}
    config["compression"].update(data["overrided_param"])
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    for wq_info in compression_ctrl.weight_quantizers.values():
        assert isinstance(wq_info.quantizer_module_ref, data["expected_weights_q"])

    for aq_info in compression_ctrl.non_weight_quantizers.values():
        assert isinstance(aq_info.quantizer_module_ref, data["expected_activations_q"])


def test_quantization_preset_with_scope_overrides():
    model = QuantizeOutputsTestModel()
    config = get_empty_config(input_sample_sizes=[2, 3, 32, 32])
    config["target_device"] = "TRIAL"
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
        "scope_overrides": {
            "weights": {
                "QuantizeOutputsTestModel/NNCFConv2d[conv5]/conv2d_0": {
                    "mode": "asymmetric",
                }
            }
        },
    }
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    for wq_info in compression_ctrl.weight_quantizers.values():
        if wq_info.affected_insertions[0].target_node_name != "QuantizeOutputsTestModel/NNCFConv2d[conv5]/conv2d_0":
            assert isinstance(wq_info.quantizer_module_ref, SymmetricQuantizer)
        else:
            assert isinstance(wq_info.quantizer_module_ref, AsymmetricQuantizer)

    for aq_info in compression_ctrl.non_weight_quantizers.values():
        assert isinstance(aq_info.quantizer_module_ref, AsymmetricQuantizer)


def test_quantization_can_be_run_with_no_data_loaders_if_zero_init_samples():
    model = BasicConvTestModel()
    # Should complete successfully even though no loaders have been registered into the config.
    _, _ = create_compressed_model_and_algo_for_test(
        model,
        NNCFConfig.from_dict(
            {
                "input_info": {"sample_size": [1, 1, 4, 4]},
                "compression": {
                    "algorithm": "quantization",
                    "initializer": {
                        "range": {"num_init_samples": 0},
                        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
                    },
                },
            }
        ),
    )


class TestHalfPrecisionModels:
    class RegularModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = torch.nn.Conv2d(1, 1, 1)
            self.conv_second = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            y = self.conv_first(x)
            y = self.conv_second(y)
            return y

    class ModelWithInternalAutocast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TestHalfPrecisionModels.RegularModel()

        def forward(self, x):
            with autocast(device_type="cuda" if x.is_cuda else "cpu"):
                y = self.model(x)
            return y

    class ModelWithManualPartialHalfPrecision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = torch.nn.Conv2d(1, 1, 1)
            self.conv_second = torch.nn.Conv2d(1, 1, 1).half()
            self.conv_third = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            y = self.conv_first(x)
            y = y.half()
            y = self.conv_second(y)
            y = y.to(torch.float32)
            y = self.conv_third(y)
            return y

    @pytest.fixture()
    def initializing_config(self):
        config = get_quantization_config_without_range_init(model_size=1)

        # Make sure that both symmetric and asymmetric quantizers appear in the model
        config["compression"]["scope_overrides"] = {
            "activations": {"{re}.*conv_first.*": {"mode": "asymmetric"}, "{re}.*conv_second.*": {"mode": "symmetric"}},
            "weights": {"{re}.*conv_first.*": {"mode": "asymmetric"}, "{re}.*conv_second.*": {"mode": "symmetric"}},
        }
        config["compression"]["initializer"] = {
            "range": {"num_init_samples": 2},
            "batchnorm_adaptation": {"num_bn_adaptation_samples": 1},
        }
        data_loader = create_ones_mock_dataloader(config)
        config = register_default_init_args(config, data_loader)
        return config

    def test_internal_autocast_model(self, initializing_config: NNCFConfig):
        model = TestHalfPrecisionModels.ModelWithInternalAutocast()
        inputs = torch.ones([1, 1, 1, 1])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            model = model.cuda()

        compressed_model, _ = create_compressed_model_and_algo_for_test(model, initializing_config)

        # Should complete successfully, including init.
        compressed_model(inputs)

    @pytest.mark.parametrize(
        "device",
        [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("cpu", marks=pytest.mark.skip(reason="CVS-86697"))],
    )
    def test_manual_partial_half_precision_model(self, initializing_config: NNCFConfig, device: str):
        model = TestHalfPrecisionModels.ModelWithManualPartialHalfPrecision()
        inputs = torch.ones([1, 1, 1, 1])

        if device == "cuda":
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                model = model.cuda()
            else:
                pytest.skip("CUDA is not available.")

        compressed_model, _ = create_compressed_model_and_algo_for_test(model, initializing_config)

        # Should complete successfully, including init.
        compressed_model(inputs)

    def test_external_autocast(self, initializing_config: NNCFConfig, use_cuda):
        model = TestHalfPrecisionModels.RegularModel()
        inputs = torch.ones([1, 1, 1, 1])
        if use_cuda:
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            inputs = inputs.cuda()
            model = model.cuda()

        compressed_model, _ = create_compressed_model_and_algo_for_test(model, initializing_config)

        with autocast(device_type="cuda" if inputs.is_cuda else "cpu"):
            # Should complete successfully.
            result = compressed_model(inputs)
            if torch.is_autocast_enabled():  # For torch <= 1.9.1 and CPU the autocast context won't have effect
                assert result.dtype == torch.float16


@pytest.mark.parametrize(
    "update_config_info, should_ignore_quantizers",
    [
        ({}, []),
        (
            {"ignored_scopes": ["LeNet/relu_1"]},
            [],  # ignoring second op in the pattern doesn't lead to exclusion of quantization first op
        ),
        (
            {"activations": {"ignored_scopes": ["LeNet/relu_1"]}},
            [],  # ignoring second op in the pattern doesn't lead to exclusion of quantization first op
        ),
        (
            {"ignored_scopes": ["LeNet/NNCFConv2d[conv2]/conv2d_0"]},
            ["LeNet/relu_0", "LeNet/NNCFConv2d[conv2]/conv2d_0"],
        ),
        ({"activations": {"ignored_scopes": ["LeNet/NNCFConv2d[conv2]/conv2d_0"]}}, ["LeNet/relu_0"]),
    ],
)
def test_activation_ignored_scope(update_config_info, should_ignore_quantizers):
    model = LeNet()
    all_quantization_names = [
        "LeNet/NNCFConv2d[conv1]/conv2d_0",
        "LeNet/NNCFConv2d[conv2]/conv2d_0",
        "LeNet/NNCFLinear[fc1]/linear_0",
        "LeNet/NNCFLinear[fc2]/linear_0",
        "LeNet/NNCFLinear[fc3]/linear_0",
        "/nncf_model_input_0",
        "LeNet/relu_0",
        "LeNet/relu_1",
        "LeNet/relu_2",
        "LeNet/relu_3",
    ]
    ref_quantization_names = list(filter(lambda x: x not in should_ignore_quantizers, all_quantization_names))
    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    config["compression"].update(update_config_info)
    train_loader = create_random_mock_dataloader(config, num_samples=10)
    config = register_default_init_args(config, train_loader)
    ctrl, _ = create_compressed_model(model, config)
    assert Counter([item.target_node_name for item in ctrl.all_quantizations]) == Counter(ref_quantization_names)


def test_sync_of_level_ranges_and_signed_parameter():
    qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        scale_shape=(1,),
        narrow_range=False,
        half_range=False,
        logarithm_scale=False,
    )

    sq = SymmetricQuantizer(qspec)
    # Check if the default values are different from the values to be loaded.
    assert sq.signed is False
    assert sq.level_low == 0

    sq.signed = True
    assert sq.signed is True
    assert sq.level_low == -8

    loaded_sq = SymmetricQuantizer(qspec)
    loaded_sq.load_state_dict(sq.state_dict())
    assert loaded_sq.signed is True
    assert loaded_sq.level_low == -8


@register_module()
class UserModuleWithAddmm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1, 1]))
        self.bias = torch.nn.Parameter(torch.ones([1, 1]))

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight)


class ModelWithUserModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_module = UserModuleWithAddmm()

    def forward(self, x):
        x = self.user_module(x)
        return x


def test_can_quantize_user_module_with_addmm():
    nncf_config = NNCFConfig.from_dict(
        {"input_info": {"sample_size": [1, 1]}, "compression": {"algorithm": "quantization"}}
    )

    train_loader = create_random_mock_dataloader(nncf_config, num_samples=10)
    nncf_config = register_default_init_args(nncf_config, train_loader)

    # Should complete successfully without exceptions:
    create_compressed_model_and_algo_for_test(ModelWithUserModule(), nncf_config)


@pytest.mark.nightly
@pytest.mark.cuda
def test_works_when_wrapped_with_dataparallel():
    if not torch.cuda.is_available() and torch.cuda.device_count() > 1:
        pytest.xfail("The executing host must have > 1 CUDA GPU in order for this test to be relevant.")

    model = SharedLayersModel().cuda()
    config = get_quantization_config_without_range_init(model_size=1)
    register_bn_adaptation_init_args(config)
    model, _ = create_compressed_model_and_algo_for_test(model, config)
    model = torch.nn.DataParallel(model)
    model(torch.ones([10, 1, 1, 1], device="cuda"))
