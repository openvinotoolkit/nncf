from typing import List
import numpy as np
import onnx
import onnxruntime as rt
import torch
from torch import nn
from nncf.torch.checkpoint_loading import load_state

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
def test_is_correct_overflow_issue_levels(num_bits, mode, scale_shape, half_range, assert_vals):
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


def helper_to_test_if_overflow_fix_was_applied(nncf_config, target_device):
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


def helper_to_test_if_overflow_fix_was_applied_only_to_first_conv_later(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for idx, quantizer in enumerate(compression_ctrl.weight_quantizers.values()):
        if idx == 0:
            assert quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
        else:
            assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access


def helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, target_device):
    model = TwoConvTestModel()
    nncf_config.update({"target_device": target_device})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for quantizer in compression_ctrl.weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access
    for quantizer in compression_ctrl.non_weight_quantizers.values():
        assert not quantizer.quantizer_module_ref._half_range  # pylint: disable=protected-access


def test_config_option_disable_overflow_fix():
    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"compression":
                            {"algorithm": "quantization",
                             "overflow_fix": "disable"}})

    for device in ['CPU', 'ANY', 'VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)

    nncf_config.update({"compression":
                            {"algorithm": "quantization",
                             "overflow_fix": "enable"}})

    for device in ['CPU', 'ANY']:
        helper_to_test_if_overflow_fix_was_applied(nncf_config, device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)

    nncf_config.update({"compression":
                            {"algorithm": "quantization",
                             "overflow_fix": "first_layer_only"}})

    for device in ['CPU', 'ANY']:
        helper_to_test_if_overflow_fix_was_applied_only_to_first_conv_later(nncf_config, device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        helper_to_test_if_overflow_fix_wasnt_applied(nncf_config, device)


def test_hw_config_overflow_fix_applied():
    nncf_config = get_config_for_export_mode(True)

    for device in ['CPU', 'ANY']:
        helper_to_test_if_overflow_fix_was_applied(nncf_config, device)

    for device in ['VPU', 'GPU', 'TRIAL']:
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
        quantizer_scale = quantizer.quantizer_module_ref.scale

        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        assert np.allclose(tensor_weight, np.array(quantized_weights))
        assert np.allclose(level_high_ratio * quantizer_scale.detach().numpy(), input_output_high)
        assert np.allclose(-2.0 * level_positive_negative_ratio * quantizer_scale.detach().numpy(),
                           input_output_low)


def are_asymmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl):
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
        quantizer_input_range = quantizer.quantizer_module_ref.input_range
        quantizer_input_low = quantizer.quantizer_module_ref.input_low

        # Quantize weights as they are exported quantized
        quantized_weights = quantizer.quantizer_module_ref(quantizer.quantized_module.weight).detach()

        assert np.allclose(tensor_weight, np.array(quantized_weights))
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
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


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
    are_asymmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def test_are_symmetric_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


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


@pytest.mark.parametrize('model', [TwoConvTestModel(), EightConvTestModel(), DepthWiseConvTestModel()])
def test_is_pytorch_output_the_same_as_onnx_qdq_overflow_fix_applied(tmp_path, model):
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


def test_is_overflow_fix_applied_model_resumed_correctly(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(False)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    compression_state = compression_ctrl.get_compression_state()
    model_state_dict = compressed_model.state_dict()
    # Must create new model as the previous one was somehow changed during create_compressed_model_and_algo_for_test()
    model = TwoConvTestModel()
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, nncf_config, compression_state=compression_state)
    load_state(compressed_model, model_state_dict, is_resume=True)
    are_symmetric_fq_nodes_are_exported_correct_with_overflow_fix(tmp_path, compression_ctrl)


def set_parameters_to_quantizer_and_get_attrs(quantizer, paramaters_to_set):
    if isinstance(quantizer, SymmetricQuantizer):
        return set_scale_to_sym_quantizer_and_get_attrs(quantizer, **paramaters_to_set)
    return set_input_low_and_input_range_to_asym_quantizer_and_get_attrs(quantizer, **paramaters_to_set)


def set_scale_to_sym_quantizer_and_get_attrs(quantizer: SymmetricQuantizer, scale: float):
    scale = np.full(quantizer.scale.size(), scale)
    levels = quantizer.levels
    level_low = quantizer.level_low
    level_high = quantizer.level_high
    input_low = scale * (level_low / level_high)
    input_range = scale - input_low
    quant_len = input_range / (levels - 1)
    quantizer.scale = nn.Parameter(torch.from_numpy(scale.astype(np.single)))
    return input_low, quant_len, levels


def set_input_low_and_input_range_to_asym_quantizer_and_get_attrs(quantizer: AsymmetricQuantizer, input_low: float,
                                                                  input_range: float):
    input_low = np.full(quantizer.input_low.size(), input_low)
    input_range = np.full(quantizer.input_low.size(), input_range)
    levels = quantizer.levels
    quant_len = input_range / (levels - 1)
    quantizer.input_low = nn.Parameter(torch.from_numpy(input_low.astype(np.single)))
    quantizer.input_range = nn.Parameter(torch.from_numpy(input_range.astype(np.single)))
    return input_low, quant_len, levels


def generate_middle_quants(size: List[int], input_low: np.ndarray, quant_len: np.ndarray, levels: np.ndarray):
    ref_weights = ([input_low + (i + 0.5) * quant_len for i in range(levels)])
    elems = np.prod(size)
    ref_weights = ref_weights * int(np.round(0.5 + elems / levels))
    ref_weights = np.reshape(np.array(ref_weights).flatten()[:elems], size, 'F')
    return torch.from_numpy(ref_weights.astype(np.single))


@pytest.mark.parametrize("quantization_mode, parameters_to_set",
                         [("symmetric", {"scale": 1.}), ("asymmetric", {"input_low": -1.,
                                                                        "input_range": 3.})])
def test_overflow_fix_quantization_export_with_middle_quants(quantization_mode, parameters_to_set):
    model = nn.Sequential(nn.Linear(in_features=128, out_features=100))
    sample_size = [1, 1, 100, 128]
    config = get_config_for_export_mode(False)
    config["compression"]["weights"] = {"mode": quantization_mode}
    config["input_info"]["sample_size"] = sample_size

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    weight_quantizer = list(compression_ctrl.weight_quantizers.values())[0].quantizer_module_ref

    input_low, quant_len, levels = set_parameters_to_quantizer_and_get_attrs(weight_quantizer, parameters_to_set)

    nncf_linear_module = list(compressed_model.get_nncf_wrapped_model())[0]
    ref_weights = generate_middle_quants(list(nncf_linear_module.weight.size()), input_low, quant_len, levels)

    nncf_linear_module.weight = nn.Parameter(ref_weights)

    compression_ctrl.export_model('linear_model.onnx')
    model_onnx = onnx.load('linear_model.onnx')

    fq_nodes = get_nodes_by_type(model_onnx, 'FakeQuantize')
    inputs = [get_all_inputs_for_graph_node(fq_node, model_onnx.graph) for fq_node in fq_nodes]
    act_weights = list(inputs[1].values())[0]

    diff = (weight_quantizer(ref_weights).detach() - act_weights).abs()

    if (diff > 1e-6).any():
        assert ((diff[diff > 1e-6] - quant_len).abs() < 1e-6).all(), 'quants completely different!'
        assert False, f'quant moved at flatten positions {torch.where(diff.flatten() > 1e-6)}'
