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

import os
from typing import List

import pytest
import torch

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationRule
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdatePaddingValue
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.quantization.adjust_padding import CalculatePaddingAdjustment
from nncf.torch.quantization.layers import SymmetricQuantizer
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import load_exported_onnx_version
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_hawq_precision_init import check_bitwidth_graph
from tests.torch.test_compressed_graph import GeneralModelDesc
from tests.torch.test_models.synthetic import MultiBranchesModel


class MultiBranchesModelDesc(GeneralModelDesc):
    NUM_WEIGHTS = 5
    NUM_ACTIVATIONS = 2

    def __init__(self, name: str):
        super().__init__(input_sample_sizes=[2, 3, 4, 4], model_name=name, model_builder=MultiBranchesModel)
        self._config = get_empty_config(input_sample_sizes=self.input_sample_sizes)
        self._config_update = {
            "compression": {
                "algorithm": "quantization",
                "scope_overrides": {
                    "activations": {"MultiBranchesModel/NNCFConv2d[conv_a]/conv2d_0": {"per_channel": True}}
                },
            }
        }
        self._hw_config = False
        self.custom_hw_config_dict = None
        self.propagation_strategy = QuantizerPropagationRule.MERGE_ALL_IN_ONE

    def requant_prop_strategy(self):
        self.propagation_strategy = QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION
        return self

    @staticmethod
    def _get_scopes():
        w_scopes = [
            "MultiBranchesModel/NNCFConv2d[conv_a]/conv2d_0|WEIGHT",
            "MultiBranchesModel/NNCFConv2d[conv_b]/conv2d_0|WEIGHT",
            "MultiBranchesModel/NNCFConv2d[conv_c]/conv2d_0|WEIGHT",
            "MultiBranchesModel/NNCFConv2d[conv_d]/conv2d_0|WEIGHT",
        ]
        a_scopes = [
            "MultiBranchesModel/NNCFConv2d[conv_a]/conv2d_0|INPUT0",
            "MultiBranchesModel/MaxPool2d[max_pool_b]/max_pool2d_0|INPUT0",
            "MultiBranchesModel/NNCFConv2d[conv_c]/conv2d_0|INPUT0",
            "MultiBranchesModel/NNCFConv2d[conv_d]/conv2d_0|INPUT0",
        ]
        return w_scopes, a_scopes

    def trial(self, num_bits_for_weights: int = 8, num_bits_for_activations: int = 8):
        self._config_update["target_device"] = "TRIAL"
        trial_config = {
            "activations": {
                "mode": "symmetric",
                "bits": num_bits_for_activations,
                "per_channel": False,
            },
            "weights": {
                "mode": "symmetric",
                "bits": num_bits_for_weights,
                "per_channel": False,
            },
        }
        self._config_update["compression"].update(trial_config)
        return self

    def npu(self):
        self._config_update["target_device"] = "NPU"
        return self

    def custom_hw(self):
        custom_hw_config_dict = {
            "target_device": "NPU",
            "config": {
                "quantization": {
                    "q4": {"bits": 4, "mode": "symmetric", "granularity": "pertensor"},
                }
            },
            "operations": [
                {"type": "Convolution", "quantization": {"activations": "q4", "weights": "q4"}},
                {
                    "type": "DepthWiseConvolution",
                    "attributes": {"adjust_padding": True},
                    "quantization": {"activations": "q4", "weights": "q4"},
                },
            ],
        }
        self.custom_hw_config_dict = custom_hw_config_dict
        # The common scope overrides conflict with the custom HW config here:
        del self._config_update["compression"]["scope_overrides"]
        return self

    def manual_precision(self, num_bits_for_weights: List[int], num_bits_for_activations: List[int]):
        scopes_factory = self._get_scopes
        w_scopes, a_scopes = scopes_factory()
        bitwidth_per_scope = list(map(list, zip(num_bits_for_weights, w_scopes)))
        bitwidth_per_scope.extend(list(map(list, zip(num_bits_for_activations, a_scopes))))
        init_config = {"initializer": {"precision": {"type": "manual", "bitwidth_per_scope": bitwidth_per_scope}}}
        self._config_update["compression"].update(init_config)
        return self

    def get_config(self):
        self._config.update(self._config_update)
        self._config["compression"].update()
        return self._config


ADJUST_PAD_DESC_LIST = [
    MultiBranchesModelDesc(name="npu_all_int8").npu(),
    MultiBranchesModelDesc(name="npu_all_weights_int8").npu().manual_precision([8, 8, 8, 8], [8, 4, 4, 4]),
    MultiBranchesModelDesc(name="npu_all_activations_int8").npu().manual_precision([8, 4, 4, 4], [8, 8, 8, 4]),
    MultiBranchesModelDesc(name="npu_bd_int8").npu().manual_precision([4, 4, 4, 4], [8, 8, 4, 8]),
    MultiBranchesModelDesc(name="npu_max_int4").npu().manual_precision([4, 4, 4, 4], [8, 4, 4, 4]),
    MultiBranchesModelDesc(name="npu_all_int8_requnt").npu().requant_prop_strategy(),
    MultiBranchesModelDesc(name="npu_all_weights_int8_requnt")
    .npu()
    .manual_precision([8, 8, 8, 8], [8, 4, 4, 4])
    .requant_prop_strategy(),
    MultiBranchesModelDesc(name="npu_all_activations_int8_requnt")
    .npu()
    .manual_precision([8, 4, 4, 4], [8, 8, 8, 4])
    .requant_prop_strategy(),
    MultiBranchesModelDesc(name="npu_bd_int8_requnt")
    .npu()
    .manual_precision([4, 4, 4, 4], [8, 8, 4, 8])
    .requant_prop_strategy(),
    MultiBranchesModelDesc(name="npu_max_int4_requnt")
    .npu()
    .manual_precision([4, 4, 4, 4], [8, 4, 4, 4])
    .requant_prop_strategy(),
    MultiBranchesModelDesc(name="custom").custom_hw(),
]


@pytest.mark.parametrize("desc", ADJUST_PAD_DESC_LIST, ids=[m.model_name for m in ADJUST_PAD_DESC_LIST])
def test_adjust_padding_on_synthetic_models(desc: MultiBranchesModelDesc, mocker, monkeypatch):
    model = desc.get_model()
    config = desc.get_config()
    register_bn_adaptation_init_args(config)

    if desc.custom_hw_config_dict:
        hw_config_from_json = mocker.patch("nncf.common.hardware.config.HWConfig.from_json")
        hw_config_from_json.return_value = PTHWConfig.from_dict(desc.custom_hw_config_dict)

    monkeypatch.setattr(QuantizerPropagationSolver, "DEFAULT_PROPAGATION_STRATEGY", desc.propagation_strategy)

    model, algo_ctrl = create_compressed_model_and_algo_for_test(model, config)

    check_bitwidth_graph(algo_ctrl, model, desc.get_dot_filename(), os.path.join("quantized", "adjust_paddings"))


def test_onnx_export_to_fake_quantize_with_adjust_pad(tmp_path):
    desc = MultiBranchesModelDesc(name="npu_max_int4").npu().manual_precision([4, 4, 4, 4], [8, 4, 4, 4])
    model = desc.get_model()
    nncf_config = desc.get_config()
    register_bn_adaptation_init_args(nncf_config)

    onnx_model_proto = load_exported_onnx_version(
        nncf_config, model, path_to_storage_dir=tmp_path, save_format="onnx_10"
    )
    num_fq = 0
    num_model_nodes = 0
    num_adjust_pad_nodes = 0
    num_other_nodes = 0

    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        if op_type == "FakeQuantize":
            num_fq += 1
        elif op_type in ["Conv", "Constant", "Relu", "MaxPool"]:
            num_model_nodes += 1
        elif op_type in ["Pad"]:
            pad_value_attr = node.attribute[2]
            assert pad_value_attr.f == 0.5
            num_adjust_pad_nodes += 1
        else:
            num_other_nodes += 1
            print(op_type)
    assert num_fq == 8
    assert num_other_nodes == 0


def test_adjust_padding_via_mixin_module(mocker):
    input_ = torch.ones([1, 1, 1, 1])
    ref_output_without_pre_ops = torch.Tensor([[[[4]]]])
    ref_output_with_update_weight = torch.Tensor([[[[3]]]])
    ref_output_with_update_weight_and_pad = torch.Tensor([[[[23]]]])

    conv = create_conv(in_channels=1, out_channels=1, kernel_size=3, weight_init=1, bias_init=2, padding=1)
    nncf_conv = NNCFConv2d.from_module(conv)
    assert nncf_conv.get_padding_value_ref().item() == 0

    act_output = nncf_conv(input_)
    assert torch.all(torch.eq(act_output, ref_output_without_pre_ops))

    uw = UpdateWeight(lambda x: torch.ones([1, 1, 3, 3]))
    nncf_conv.register_pre_forward_operation(uw)
    act_output = nncf_conv(input_)
    assert torch.all(torch.eq(act_output, ref_output_with_update_weight))

    quantizer_stub = mocker.MagicMock(spec=SymmetricQuantizer)
    quantizer_stub.scale = torch.Tensor([4])
    quantizer_stub.eps = torch.Tensor([1])
    ap = CalculatePaddingAdjustment(quantizer_stub)
    upv = UpdatePaddingValue(ap)
    nncf_conv.register_pre_forward_operation(upv)
    act_output = nncf_conv(input_)
    assert nncf_conv.get_padding_value_ref().item() == 2.5
    assert torch.all(torch.eq(act_output, ref_output_with_update_weight_and_pad))
