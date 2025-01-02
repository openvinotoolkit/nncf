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

import numpy as np
import onnx
import onnxruntime as rt
import pytest
import torch
from torch import nn

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_all_inputs_for_graph_node
from tests.torch.helpers import get_nodes_by_type
from tests.torch.quantization.test_onnx_export import get_config_for_export_mode


@pytest.mark.parametrize(
    "num_bits, mode, scale_shape, half_range, assert_vals",
    [
        (8, QuantizationMode.SYMMETRIC, (1, 2, 3, 4), True, (128, -64, 63)),
        (8, QuantizationMode.ASYMMETRIC, (1, 2, 3, 4), True, (128, 0, 127)),
        (7, QuantizationMode.SYMMETRIC, (1, 2, 3, 4), True, (64, -32, 31)),
        (4, QuantizationMode.SYMMETRIC, (1, 1, 1, 1), True, (8, -4, 3)),
        (8, QuantizationMode.SYMMETRIC, (1, 1, 1, 1), True, (128, -64, 63)),
        (8, QuantizationMode.SYMMETRIC, (1, 2, 3, 8), False, (256, -128, 127)),
    ],
)
def test_is_correct_overflow_issue_levels(num_bits, mode, scale_shape, half_range, assert_vals):
    qspec = PTQuantizerSpec(
        num_bits=num_bits,
        mode=mode,
        signedness_to_force=True,
        narrow_range=False,
        scale_shape=scale_shape,
        logarithm_scale=False,
        half_range=half_range,
        is_quantized_on_export=True,
    )

    quantizer = SymmetricQuantizer(qspec) if mode == QuantizationMode.SYMMETRIC else AsymmetricQuantizer(qspec)

    assert quantizer._half_range == half_range
    assert quantizer.levels == assert_vals[0]
    assert quantizer.level_low == assert_vals[1]
    assert quantizer.level_high == assert_vals[2]


def helper_to_test_if_overflow_fix_was_applied(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for quantizer in compression_ctrl.weight_quantizers.values():
        assert quantizer.quantizer_module_ref._half_range
        assert quantizer.quantizer_module_ref.levels == 128
        assert quantizer.quantizer_module_ref.level_low == -64
        assert quantizer.quantizer_module_ref.level_high == 63

    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range


def helper_to_test_if_overflow_fix_was_applied_only_to_first_conv_later(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for idx, quantizer in enumerate(compression_ctrl.weight_quantizers.values()):
        if idx == 0:
            assert quantizer.quantizer_module_ref._half_range
        else:
            assert not quantizer.quantizer_module_ref._half_range
    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range


def helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for quantizer in compression_ctrl.weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range
    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range


def test_config_option_disable_overflow_fix():
    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"compression": {"algorithm": "quantization", "overflow_fix": "disable"}})

    for device in ["CPU", "ANY", "NPU", "GPU", "TRIAL"]:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)

    nncf_config.update({"compression": {"algorithm": "quantization", "overflow_fix": "enable"}})

    for device in ["CPU", "ANY"]:
        helper_to_test_if_overflow_fix_was_applied(nncf_config, device)

    for device in ["NPU", "GPU", "TRIAL"]:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)

    nncf_config.update({"compression": {"algorithm": "quantization", "overflow_fix": "first_layer_only"}})

    for device in ["CPU", "ANY"]:
        helper_to_test_if_overflow_fix_was_applied_only_to_first_conv_later(nncf_config, device)

    for device in ["NPU", "GPU", "TRIAL"]:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)


def test_hw_config_overflow_fix_applied():
    nncf_config = get_config_for_export_mode(True)

    for device in ["CPU", "ANY"]:
        helper_to_test_if_overflow_fix_was_applied(nncf_config, device)

    for device in ["NPU", "GPU", "TRIAL"]:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)


class EightConvTestModel(nn.Module):
    def __init__(self, in_out_ch=((1, 3), (3, 5), (5, 7), (7, 10))):
        super().__init__()
        self.features = []
        self.features.append(create_conv(*in_out_ch[0], 2, -1, -2))
        self.features.append(nn.BatchNorm2d(in_out_ch[0][1]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*in_out_ch[1], 5, 1, 1))
        self.features.append(nn.BatchNorm2d(in_out_ch[1][1]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*in_out_ch[2], 1, 2, 2))
        self.features.append(nn.BatchNorm2d(in_out_ch[2][1]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*in_out_ch[3], 9, -1, 0))
        self.features.append(nn.BatchNorm2d(in_out_ch[3][1]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*reversed(in_out_ch[3]), 3, 0, 1))
        self.features.append(nn.BatchNorm2d(in_out_ch[3][0]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*reversed(in_out_ch[2]), 1, -1, 9))
        self.features.append(nn.BatchNorm2d(in_out_ch[2][0]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*reversed(in_out_ch[1]), 2, 10, 1))
        self.features.append(nn.BatchNorm2d(in_out_ch[1][0]))
        self.features.append(nn.ReLU())
        self.features.append(create_conv(*reversed(in_out_ch[0]), 1, 1, 1))
        self.features.append(nn.BatchNorm2d(in_out_ch[0][0]))
        self.features.append(nn.ReLU())
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)


class DepthWiseConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 3, 3, groups=1))
        self.features.append(nn.Conv2d(3, 30, 3, groups=3))
        self.features.append(nn.Conv2d(30, 1, 3))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)


def are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl):
    level_high = 63
    level_low = -64
    levels = 128
    # Update scale tensors in the model
    quantizers = compression_ctrl.weight_quantizers.values()
    with torch.no_grad():
        for quantizer in quantizers:
            assert quantizer.quantizer_module_ref.levels == levels
            assert quantizer.quantizer_module_ref._half_range
            assert quantizer.quantizer_module_ref.level_low == level_low
            assert quantizer.quantizer_module_ref.level_high == level_high
            quantizer.quantizer_module_ref.scale = torch.nn.Parameter(
                5 * torch.rand_like(quantizer.quantizer_module_ref.scale, dtype=torch.float32, requires_grad=True)
            )

    onnx_checkpoint_path = str(tmp_path / "model.onnx")
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=["input"])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    fq_nodes = get_nodes_by_type(onnx_model, "FakeQuantize")

    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in fq_nodes]

    level_high_ratio = (2 * level_high + 1) / level_high
    level_positive_negative_ratio = abs(level_low / level_high)

    for quantizer, fq_parametres in zip(quantizers, inputs[1::2]):
        tensor_weight, input_output_low, input_output_high = list(fq_parametres.values())
        quantizer_scale = quantizer.quantizer_module_ref.scale

        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        assert np.allclose(tensor_weight, np.array(quantized_weights))
        assert np.allclose(level_high_ratio * quantizer_scale.detach().numpy(), input_output_high)
        assert np.allclose(-2.0 * level_positive_negative_ratio * quantizer_scale.detach().numpy(), input_output_low)


def are_asymmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl):
    level_high = 127
    level_low = 0
    levels = 128
    # Update scale tensors in the model
    quantizers = compression_ctrl.weight_quantizers.values()
    with torch.no_grad():
        for quantizer in quantizers:
            assert quantizer.quantizer_module_ref.levels == levels
            assert quantizer.quantizer_module_ref._half_range
            assert quantizer.quantizer_module_ref.level_low == level_low
            assert quantizer.quantizer_module_ref.level_high == level_high
            quantizer.quantizer_module_ref.input_range = torch.nn.Parameter(
                5 * torch.rand_like(quantizer.quantizer_module_ref.input_range, dtype=torch.float32, requires_grad=True)
            )

    onnx_checkpoint_path = str(tmp_path / "model.onnx")
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=["input"])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    fq_nodes = get_nodes_by_type(onnx_model, "FakeQuantize")

    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in fq_nodes]

    level_high_ratio = (2 * level_high + 1) / level_high
    for quantizer, fq_parametres in zip(quantizers, inputs[1::2]):
        tensor_weight, input_output_low, input_output_high = list(fq_parametres.values())
        quantizer_input_range = quantizer.quantizer_module_ref.input_range
        quantizer_input_low = quantizer.quantizer_module_ref.input_low

        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        assert np.allclose(tensor_weight, np.array(quantized_weights))
        assert np.allclose(level_high_ratio * quantizer_input_range.detach().numpy(), input_output_high)
        assert np.allclose(quantizer_input_low.detach().numpy(), input_output_low)


def test_are_symmetric_fq_exported_depthwise_per_channel_weights_tensors_clipped(tmp_path):
    model = DepthWiseConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {"sample_size": [1, 1, 20, 20]}})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def test_are_asymmetric_fq_exported_depthwise_per_channel_weights_tensors_clipped(tmp_path):
    model = DepthWiseConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {"sample_size": [1, 1, 20, 20]}})
    nncf_config.update(
        {
            "compression": {
                "algorithm": "quantization",
                "export_to_onnx_standard_ops": False,
                "weights": {"mode": "asymmetric"},
            }
        }
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_asymmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def test_are_symmetric_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {"sample_size": [1, 1, 20, 20]}})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def test_are_assymetric_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {"sample_size": [1, 1, 20, 20]}})
    nncf_config.update(
        {
            "compression": {
                "algorithm": "quantization",
                "export_to_onnx_standard_ops": False,
                "weights": {"mode": "asymmetric"},
            }
        }
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_asymmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def test_are_qdq_exported_per_tensor_weights_tensors_clipped(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(True)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    # Set scale tensors
    first_scale_tensor = torch.tensor((0.1, 0.1), dtype=torch.float32, requires_grad=True)
    second_scale_tensor = torch.tensor(3000, dtype=torch.float32, requires_grad=True)

    # Update scale tensors in the model
    first_quantizer, second_quantizer = compression_ctrl.weight_quantizers.values()

    first_quantizer.quantizer_module_ref.scale = torch.nn.Parameter(first_scale_tensor)
    second_quantizer.quantizer_module_ref.scale = torch.nn.Parameter(second_scale_tensor)

    for quantizer in [first_quantizer, second_quantizer]:
        assert quantizer.quantizer_module_ref.levels == 128
        assert quantizer.quantizer_module_ref.level_low == -64
        assert quantizer.quantizer_module_ref.level_high == 63
        assert quantizer.quantizer_module_ref._half_range

    onnx_checkpoint_path = str(tmp_path / "model.onnx")
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=["input"], save_format="onnx_13")

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    quantize_nodes = get_nodes_by_type(onnx_model, "QuantizeLinear")

    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in quantize_nodes]

    level_high_ratio = 127.0 / 63.0
    level_positive_negative_ratio = 64.0 / 63.0

    for quantizer, onnx_q_parametres in zip([first_quantizer, second_quantizer], inputs[1::2]):
        onnx_tensor_weight, onnx_q_scale, onnx_zero_level = list(onnx_q_parametres.values())
        quantizer_scale = quantizer.quantizer_module_ref.scale.detach().numpy()

        onnx_input_output_low = -128 * onnx_q_scale + onnx_zero_level
        onnx_input_output_high = 127 * onnx_q_scale + onnx_zero_level

        if quantizer_scale.shape:
            quantizer_scale = quantizer_scale[0]

        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        assert np.allclose(onnx_tensor_weight, np.array(quantized_weights))
        assert np.allclose(level_high_ratio * quantizer_scale, onnx_input_output_high)
        assert np.allclose(-2.0 * level_positive_negative_ratio * quantizer_scale, onnx_input_output_low)


@pytest.mark.parametrize("model", [TwoConvTestModel(), EightConvTestModel(), DepthWiseConvTestModel()])
def test_is_pytorch_output_the_same_as_onnx_qdq_overflow_fix_applied(tmp_path, model):
    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"input_info": {"sample_size": [1, 1, 20, 20]}})

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    onnx_checkpoint_path = str(tmp_path / "model.onnx")
    compression_ctrl.export_model(onnx_checkpoint_path, save_format="onnx_13")
    input_tensors = [
        np.random.normal(size=[1, 1, 20, 20]),
        np.random.uniform(size=[1, 1, 20, 20]),
        100 * np.random.normal(size=[1, 1, 20, 20]),
        100 * np.random.uniform(size=[1, 1, 20, 20]),
    ]
    for input_tensor in input_tensors:
        torch_input = torch.tensor(input_tensor, dtype=torch.float32)

        with torch.no_grad():
            torch_out = compressed_model(torch_input)

        # ONNXRuntime
        sess = rt.InferenceSession(onnx_checkpoint_path)
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: input_tensor.astype(np.float32)})[0]

        assert np.allclose(torch_out.numpy(), onnx_out, rtol=1e-5, atol=1e-3)


def test_is_overflow_fix_applied_model_resumed_correctly(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    compression_state = compression_ctrl.get_compression_state()
    model_state_dict = compressed_model.state_dict()
    # Must create new model as the previous one was somehow changed during create_compressed_model_and_algo_for_test()
    model = TwoConvTestModel()
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, nncf_config, compression_state=compression_state
    )
    load_state(compressed_model, model_state_dict, is_resume=True)
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)
