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

import torch

from nncf.dynamic_graph.graph_builder import ModelInputInfo
from nncf.dynamic_graph.patch_pytorch import MODEL_INPUT_OP_NAME
from nncf.hw_config import HWConfig
from nncf.nncf_network import  NNCFNetwork
from nncf.quantization.algo import QuantizationBuilder, QuantizerSetupType, QuantizationController
from nncf.quantization.layers import QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer, BaseQuantizer

from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init


class ModelForHWConfigTest(torch.nn.Module):
    def __init__(self, with_gelu=False):
        super().__init__()
        self.with_gelu = with_gelu
        self.conv2d = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x_):
        if self.with_gelu:
            x_ = torch.nn.functional.gelu(x_)
        x_ = self.conv2d(x_)
        x_ = x_.matmul(x_)
        return x_


class TestHWConfigRules:
    @staticmethod
    def get_model_and_ctrl_with_applied_hw_config_quantization(model: torch.nn.Module, hw_config_dict: dict):
        nncf_config = get_quantization_config_without_range_init(model_size=1)
        nncf_config["compression"].update({"quantize_inputs": False})

        net = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 2, 1, 1])])
        qbuilder = QuantizationBuilder(nncf_config["compression"], should_init=False)
        qbuilder.hw_config = HWConfig.from_dict(hw_config_dict)
        qbuilder.quantizer_setup_type = QuantizerSetupType.PROPAGATION_BASED
        net = qbuilder.apply_to(net)
        ctrl = net.commit_compression_changes()
        return net, ctrl

    @staticmethod
    def quantizer_has_default_config(quantizer: BaseQuantizer) -> bool:
        default_qconfig = QuantizationBuilder.DEFAULT_QUANTIZER_CONFIG
        is_ok = True
        is_ok &= (quantizer.num_bits == default_qconfig.bits)
        is_ok &= (quantizer.per_channel == default_qconfig.per_channel)
        if default_qconfig.signedness_to_force is not None:
            is_ok &= (quantizer.signed == default_qconfig.signedness_to_force)
        is_ok &= isinstance(quantizer,
                            SymmetricQuantizer if default_qconfig.mode == QuantizationMode.SYMMETRIC else
                            AsymmetricQuantizer)
        return is_ok

    @staticmethod
    def get_quantizer_module_after_op_name(op_name: str, ctrl: QuantizationController) -> BaseQuantizer:
        input_matches = list(filter(lambda x: x.ia_op_exec_context.operator_name == op_name,
                                    ctrl.non_weight_quantizers.keys()))
        assert len(input_matches) == 1
        act_quant_key = input_matches[0]
        act_quantizer_ref = ctrl.non_weight_quantizers[act_quant_key].quantizer_module_ref
        return act_quantizer_ref

    def test_missing_ir_op_results_in_fp32(self):
        hw_config_dict = {
            "target_device": "test",
            "config": {
                "quantization": {
                    "q8_a": {
                        "bits": 8,
                        "mode": [
                            "symmetric",
                            "asymmetric"
                        ],
                        "granularity": "pertensor"
                    },
                }
            },
            "operations": [
                {
                    "type": "MatMul",
                    "quantization": {
                        "activations": "q8_a",
                        "weights": "q8_a"
                    }
                },
            ]
        }

        _, ctrl = self.get_model_and_ctrl_with_applied_hw_config_quantization(ModelForHWConfigTest(with_gelu=False),
                                                                              hw_config_dict)
        assert len(ctrl.weight_quantizers) == 0  # Conv2d weights remain unquantized
        assert len(ctrl.non_weight_quantizers) == 1  # Only the matmul input is quantized

        key = next(iter(ctrl.non_weight_quantizers.keys()))
        # Corresponds to a quantizer AFTER conv2d, i.e. matmul input quantizer
        assert key.ia_op_exec_context.operator_name == "conv2d"

    def test_missing_non_ir_op_results_in_default_qconf_list(self):
        # GELU is the non-IR op here, adjust if this no longer reflects reality
        hw_config_dict = {
            "target_device": "test",
            "config": {
                "quantization": {
                    "q4_a": {
                        "bits": 4,
                        "mode": [
                            "symmetric",
                            "asymmetric"
                        ],
                        "granularity": "pertensor"
                    },
                }
            },
            "operations": [
                {
                    "type": "MatMul",
                    "quantization": {
                        "activations": "q4_a",
                        "weights": "q4_a"
                    },
                },
                {

                    "type": "Convolution",
                    "quantization": {
                        "activations": "q4_a",
                        "weights": "q4_a"
                    }
                },
            ]
        }

        _, ctrl = self.get_model_and_ctrl_with_applied_hw_config_quantization(ModelForHWConfigTest(with_gelu=True),
                                                                              hw_config_dict)
        assert len(ctrl.weight_quantizers) == 1  # Conv2d weights quantized
        assert len(ctrl.non_weight_quantizers) == 3  # GELU input, conv2d input, matmul input (single in this case)

        w_key = next(iter(ctrl.weight_quantizers.keys()))
        assert str(w_key.scope) == "ModelForHWConfigTest/NNCFConv2d[conv2d]"

        gelu_input_act_quantizer_ref = self.get_quantizer_module_after_op_name(MODEL_INPUT_OP_NAME, ctrl)
        assert self.quantizer_has_default_config(gelu_input_act_quantizer_ref)

    def test_unspecified_quantization_for_fundamentally_quantizable_op_results_in_default_qconfig(self):
        hw_config_dict = {  # Only the MatMul will receive a default config here (8 bit symmetric per-tensor)
            "target_device": "test",
            "config": {
                "quantization": {
                    "q4_a": {
                        "bits": 4,
                        "mode": [
                            "symmetric",
                            "asymmetric"
                        ],
                        "granularity": "pertensor"
                    },
                }
            },
            "operations": [
                {
                    "type": "MatMul"
                },
                {

                    "type": "Convolution",
                    "quantization": {
                        "activations": "q4_a",
                        "weights": "q4_a"
                    }
                },
            ]
        }

        _, ctrl = self.get_model_and_ctrl_with_applied_hw_config_quantization(ModelForHWConfigTest(with_gelu=False),
                                                                              hw_config_dict)
        assert len(ctrl.weight_quantizers) == 1  # Conv2d weights quantized
        conv2d_weight_quantizer_ref = list(ctrl.weight_quantizers.values())[0]
        assert not self.quantizer_has_default_config(conv2d_weight_quantizer_ref)

        assert len(ctrl.non_weight_quantizers) == 2  # Conv2d input, matmul input
        matmul_input_matches = list(filter(lambda x: x.ia_op_exec_context.operator_name == "conv2d",
                                           ctrl.non_weight_quantizers.keys()))

        assert len(matmul_input_matches) == 1
        matmul_quantizer_ref = ctrl.non_weight_quantizers[matmul_input_matches[0]].quantizer_module_ref
        assert self.quantizer_has_default_config(matmul_quantizer_ref)

        non_matmul_input_matches = list(filter(lambda x: x.ia_op_exec_context.operator_name != "conv2d",
                                               ctrl.non_weight_quantizers.keys()))
        for quantizer_id in non_matmul_input_matches:
            quantizer_ref = ctrl.non_weight_quantizers[quantizer_id].quantizer_module_ref
            assert not self.quantizer_has_default_config(quantizer_ref)

    def test_unspecified_quantization_for_weighted_op_results_in_default_qconf_list_for_weights(self):
        hw_config_dict = {
            "target_device": "test",
            "config": {
                "quantization": {
                    "q4_a": {
                        "bits": 4,
                        "mode": [
                            "symmetric",
                            "asymmetric"
                        ],
                        "granularity": "pertensor"
                    },
                }
            },
            "operations": [
                {
                    "type": "MatMul"
                },
                {
                    "type": "Convolution"
                },
            ]
        }

        _, ctrl = self.get_model_and_ctrl_with_applied_hw_config_quantization(ModelForHWConfigTest(with_gelu=False),
                                                                              hw_config_dict)
        assert len(ctrl.weight_quantizers) == 1  # Conv2d weights quantized with default config
        assert len(ctrl.non_weight_quantizers) == 2  # All inputs are quantized.
        for quantizer_ref in ctrl.all_quantizations.values():
            assert self.quantizer_has_default_config(quantizer_ref)
