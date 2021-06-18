import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn

from nncf.torch.quantization.layers import PTQuantizerSpec, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer
from tests.torch.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test, create_conv, \
    get_nodes_by_type, get_all_inputs_for_graph_node
from tests.torch.quantization.test_onnx_export import get_config_for_export_mode

import pytest


@pytest.mark.parametrize('num_bits, mode, scale_shape, half_range, assert_vals',
                         [(8, QuantizationMode.SYMMETRIC, (1, 2, 3, 4), True, (128, -64, 63)),
                          (8, QuantizationMode.ASYMMETRIC, (1, 2, 3, 4), True, (128, 0, 127)),
                          (7, QuantizationMode.SYMMETRIC, (1, 2, 3, 4), True, (64, -32, 31)),
                          (4, QuantizationMode.SYMMETRIC, (1, 1, 1, 1), True, (8, -4, 3)),
                          (8, QuantizationMode.SYMMETRIC, (1, 1, 1, 1), True, (128, -64, 63)),
                          (8, QuantizationMode.SYMMETRIC, (1, 2, 3, 8), False, (256, -128, 127))
                          ])
def test_is_correct_saturation_issue_levels(num_bits, mode, scale_shape, half_range, assert_vals):
    qspec = PTQuantizerSpec(
        num_bits=num_bits,
        mode=mode,
        signedness_to_force=True,
        narrow_range=False,
        scale_shape=scale_shape,
        logarithm_scale=False,
        half_range=half_range)

    quantizer = SymmetricQuantizer(qspec) if mode == QuantizationMode.SYMMETRIC else AsymmetricQuantizer(qspec)

    assert quantizer._half_range == half_range  # pylint: disable=protected-access
    assert quantizer.levels == assert_vals[0]
    assert quantizer.level_low == assert_vals[1]
    assert quantizer.level_high == assert_vals[2]


def helper_to_test_if_saturation_fix_was_applied(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for quantizer in compression_ctrl.weight_quantizers.values():
        assert quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
        assert quantizer.quantizer_module_ref.levels == 128
        assert quantizer.quantizer_module_ref.level_low == -64
        assert quantizer.quantizer_module_ref.level_high == 63

    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access


def helper_to_test_if_saturation_fix_wasnt_applied(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for quantizer in compression_ctrl.weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access


def test_config_option_disable_saturation_fix():
    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"compression":
                            {"algorithm": "quantization",
                             "disable_saturation_fix": True}})

    for device in ['CPU', 'ANY', 'VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_saturation_fix_wasnt_applied(nncf_config, device)

    nncf_config.update({"compression":
                            {"algorithm": "quantization",
                             "disable_saturation_fix": False}})

    for device in ['CPU', 'ANY']:
        helper_to_test_if_saturation_fix_was_applied(nncf_config, device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_saturation_fix_wasnt_applied(nncf_config, device)


def test_hw_config_saturation_fix_applied():
    nncf_config = get_config_for_export_mode(True)

    for device in ['CPU', 'ANY']:
        helper_to_test_if_saturation_fix_was_applied(nncf_config, device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_saturation_fix_wasnt_applied(nncf_config, device)


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


def are_symmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl):
    level_high = 63
    level_low = -64
    levels = 128
    # Update scale tensors in the model
    quantizers = compression_ctrl.weight_quantizers.values()
    with torch.no_grad():
        for quantizer in quantizers:
            assert quantizer.quantizer_module_ref.levels == levels
            assert quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
            assert quantizer.quantizer_module_ref.level_low == level_low
            assert quantizer.quantizer_module_ref.level_high == level_high
            quantizer.quantizer_module_ref.scale = torch.nn.Parameter(
                5 * torch.rand_like(quantizer.quantizer_module_ref.scale,
                                    dtype=torch.float32, requires_grad=True))

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    fq_nodes = get_nodes_by_type(onnx_model, 'FakeQuantize')
    # pylint:disable=no-member
    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in fq_nodes]

    level_high_ratio = (2 * level_high + 1) / level_high
    level_positive_negative_ratio = abs(level_low / level_high)

    for quantizer, fq_parametres in zip(quantizers, inputs[1::2]):
        tensor_weight, input_output_low, input_output_high = list(fq_parametres.values())
        quantizer_weight, quantizer_scale = quantizer.quantized_module.weight.detach().numpy(), \
                                            quantizer.quantizer_module_ref.scale

        input_low = -level_positive_negative_ratio * quantizer_scale.detach().numpy()
        input_high = quantizer_scale.detach().numpy()
        # Clamp weight tensors as we do in exporting
        if quantizer_weight.shape[0] > 1:
            for i in range(quantizer_weight.shape[0]):
                try:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i], a_max=input_high[i])
                except TypeError:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i].item(),
                                                  a_max=input_high[i].item())
        else:
            quantizer_weight = np.clip(quantizer_weight, a_min=input_low.item(), a_max=input_high.item())

        assert np.allclose(tensor_weight, quantizer_weight)
        assert np.allclose(level_high_ratio * quantizer_scale.detach().numpy(), input_output_high)
        assert np.allclose(-2.0 * level_positive_negative_ratio * quantizer_scale.detach().numpy(),
                           input_output_low)


def are_asymmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl):
    level_high = 127
    level_low = 0
    levels = 128
    # Update scale tensors in the model
    quantizers = compression_ctrl.weight_quantizers.values()
    with torch.no_grad():
        for quantizer in quantizers:
            assert quantizer.quantizer_module_ref.levels == levels
            assert quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
            assert quantizer.quantizer_module_ref.level_low == level_low
            assert quantizer.quantizer_module_ref.level_high == level_high
            quantizer.quantizer_module_ref.input_range = torch.nn.Parameter(
                5 * torch.rand_like(quantizer.quantizer_module_ref.input_range,
                                    dtype=torch.float32, requires_grad=True))

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    fq_nodes = get_nodes_by_type(onnx_model, 'FakeQuantize')
    # pylint:disable=no-member
    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in fq_nodes]

    level_high_ratio = (2 * level_high + 1) / level_high
    for quantizer, fq_parametres in zip(quantizers, inputs[1::2]):
        tensor_weight, input_output_low, input_output_high = list(fq_parametres.values())
        quantizer_weight = quantizer.quantized_module.weight.detach().numpy()
        quantizer_input_range = quantizer.quantizer_module_ref.input_range
        quantizer_input_low = quantizer.quantizer_module_ref.input_low

        input_low = quantizer_input_low.detach().numpy()
        input_high = quantizer_input_range.detach().numpy()
        # Clamp weight tensors as we do in exporting
        if quantizer_weight.shape[0] > 1:
            for i in range(quantizer_weight.shape[0]):
                try:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i], a_max=input_high[i])
                except TypeError:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i].item(),
                                                  a_max=input_high[i].item())
        else:
            quantizer_weight = np.clip(quantizer_weight, a_min=input_low.item(), a_max=input_high.item())

        assert np.allclose(tensor_weight, quantizer_weight)
        assert np.allclose(level_high_ratio * quantizer_input_range.detach().numpy(), input_output_high)
        assert np.allclose(quantizer_input_low.detach().numpy(),
                           input_output_low)


def test_are_symmetric_fq_exported_depthwise_per_channel_weights_tensors_clipped(tmp_path):
    model = DepthWiseConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_symmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl)


def test_are_asymmetric_fq_exported_depthwise_per_channel_weights_tensors_clipped(tmp_path):
    model = DepthWiseConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    nncf_config.update({"compression": {
        "algorithm": "quantization",
        "export_to_onnx_standard_ops": False,
        "weights": {
            "mode": "asymmetric"
        },
    }
    })
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_asymmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl)


def test_are_symmetric_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_symmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl)


def test_are_assymetric_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    nncf_config.update({"compression": {
        "algorithm": "quantization",
        "export_to_onnx_standard_ops": False,
        "weights": {
            "mode": "asymmetric"
        },
    }
    })
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_asymmetric_fq_nodes_are_exported_correct_with_saturation_fix(tmp_path, compression_ctrl)


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
        assert quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    quantize_nodes = get_nodes_by_type(onnx_model, 'QuantizeLinear')
    # pylint:disable=no-member
    inputs = [get_all_inputs_for_graph_node(fq_node, onnx_model.graph) for fq_node in quantize_nodes]

    level_high_ratio = 127. / 63.
    level_positive_negative_ratio = 64. / 63.

    for quantizer, onnx_q_parametres in zip([first_quantizer, second_quantizer], inputs[1::2]):
        onnx_tensor_weight, onnx_q_scale, onnx_zero_level = list(onnx_q_parametres.values())
        quantizer_weight, quantizer_scale = quantizer.quantized_module.weight.detach().numpy(), \
                                            quantizer.quantizer_module_ref.scale

        quantizer_scale = quantizer_scale.detach().numpy()
        input_low = -level_positive_negative_ratio * quantizer_scale
        input_high = quantizer_scale
        onnx_input_output_low = -128 * onnx_q_scale + onnx_zero_level
        onnx_input_output_high = 127 * onnx_q_scale + onnx_zero_level

        # Clamp weight tensors as we do in exporting
        if quantizer_weight.shape[0] > 1:
            for i in range(quantizer_weight.shape[0]):
                try:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i], a_max=input_high[i])
                except IndexError:
                    quantizer_weight[i] = np.clip(quantizer_weight[i], a_min=input_low[i].item(),
                                                  a_max=input_high[i].item())
        else:
            quantizer_weight = np.clip(quantizer_weight, a_min=input_low.item(), a_max=input_high.item())

        if quantizer_scale.shape:
            quantizer_scale = quantizer_scale[0]
        assert np.allclose(onnx_tensor_weight, quantizer_weight)
        assert np.allclose(level_high_ratio * quantizer_scale, onnx_input_output_high)
        assert np.allclose(-2.0 * level_positive_negative_ratio * quantizer_scale, onnx_input_output_low)


@pytest.mark.parametrize('model', [TwoConvTestModel(), EightConvTestModel(), DepthWiseConvTestModel()])
def test_is_pytorch_output_the_same_as_onnx_qdq_saturation_fix_applied(tmp_path, model):
    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path)
    input_tensors = [np.random.normal(size=[1, 1, 20, 20]), np.random.uniform(size=[1, 1, 20, 20]),
                     100 * np.random.normal(size=[1, 1, 20, 20]), 100 * np.random.uniform(size=[1, 1, 20, 20])]
    for input_tensor in input_tensors:
        torch_input = torch.tensor(input_tensor, dtype=torch.float32)

        with torch.no_grad():
            torch_out = compressed_model(torch_input)

        # ONNXRuntime
        sess = rt.InferenceSession(onnx_checkpoint_path)
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: input_tensor.astype(np.float32)})[0]

        assert np.allclose(torch_out.numpy(), onnx_out, rtol=1e-5, atol=1e-3)
