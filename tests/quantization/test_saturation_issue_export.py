import numpy as np
import torch
import onnx
from onnx import numpy_helper
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_onnx_export import get_config_for_export_mode
from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer


def test_is_correct_saturation_issue_levels():
    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert quantizer.is_saturation_fix
    assert quantizer.levels == 128
    assert quantizer.level_low == -64
    assert quantizer.level_high == 63

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = AsymmetricQuantizer(q_config)

    assert quantizer.is_saturation_fix
    assert quantizer.levels == 128
    assert quantizer.level_low == 0
    assert quantizer.level_high == 127

    q_config = QuantizerConfig(
        bits=7,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=True,
        is_weights=True,
        input_shape=[3, 32, 32],
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert quantizer.is_saturation_fix
    assert quantizer.levels == 64
    assert quantizer.level_low == -32
    assert quantizer.level_high == 31

    q_config = QuantizerConfig(
        bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert quantizer.is_saturation_fix
    assert quantizer.levels == 8
    assert quantizer.level_low == -4
    assert quantizer.level_high == 3

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=False,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert quantizer.is_saturation_fix
    assert quantizer.levels == 128
    assert quantizer.level_low == -64
    assert quantizer.level_high == 63

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=False)

    quantizer = SymmetricQuantizer(q_config)

    assert not quantizer.is_saturation_fix
    assert quantizer.levels == 256
    assert quantizer.level_low == -128
    assert quantizer.level_high == 127


def test_hw_config_saturation_fix_applied():
    nncf_config = get_config_for_export_mode(True)

    # Test CPU, ANY device in which we use saturation issue
    def test_with_saturation_helper(target_device):
        model = TwoConvTestModel()
        nncf_config.update({"target_device": target_device})

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

        for quantizer in compression_ctrl.weight_quantizers.values():
            assert quantizer.quantizer_module_ref.is_saturation_fix
            assert quantizer.quantizer_module_ref.levels == 128
            assert quantizer.quantizer_module_ref.level_low == -64
            assert quantizer.quantizer_module_ref.level_high == 63

        for quantizer in compression_ctrl.non_weight_quantizers.values():
            assert not quantizer.quantizer_module_ref.is_saturation_fix

    # Test other devices in which we don't use saturation issue
    def test_without_saturation_helper(target_device):
        model = TwoConvTestModel()
        nncf_config.update({"target_device": target_device})

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

        for quantizer in compression_ctrl.weight_quantizers.values():
            assert not quantizer.quantizer_module_ref.is_saturation_fix
        for quantizer in compression_ctrl.non_weight_quantizers.values():
            assert not quantizer.quantizer_module_ref.is_saturation_fix

    for device in ['CPU', 'ANY']:
        test_with_saturation_helper(device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        test_without_saturation_helper(device)


def test_are_onnx_exported_weights_tensors_clipped(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(True)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # Set scale tensors
    first_scale_tensor = torch.tensor((0.5, 0.5), dtype=torch.float32, requires_grad=True)
    second_scale_tensor = torch.tensor(3000, dtype=torch.float32, requires_grad=True)

    # Update scale tensors in the model
    first_quantizer, second_quantizer = compression_ctrl.weight_quantizers.values()

    first_quantizer.quantizer_module_ref.scale = torch.nn.Parameter(first_scale_tensor)
    second_quantizer.quantizer_module_ref.scale = torch.nn.Parameter(second_scale_tensor)

    # Set weight tensors bigger then scale tensor
    first_quantizer.quantized_module.weight.data = 50 * torch.ones_like(
        first_quantizer.quantized_module.weight)
    second_quantizer.quantized_module.weight.data = 2000 * torch.ones_like(
        second_quantizer.quantized_module.weight)

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    for node in onnx_model.graph.node:
        try:
            if node.op_type == 'Constant':
                if node.attribute[0].t.dims[0] == 2 and node.attribute[0].t.dims[1] == 1 and \
                        node.attribute[0].t.dims[2] == 2 and node.attribute[0].t.dims[3] == 2:
                    first_conv_weight_tensor = numpy_helper.to_array(node.attribute[0].t)
                elif node.attribute[0].t.dims[0] == 1 and node.attribute[0].t.dims[1] == 2 and \
                        node.attribute[0].t.dims[2] == 3 and node.attribute[0].t.dims[3] == 3:
                    second_conv_weight_tensor = numpy_helper.to_array(node.attribute[0].t)
        except (AttributeError, IndexError):
            continue

    assert np.max(first_conv_weight_tensor) <= np.max(first_scale_tensor.detach().numpy())
    assert np.min(first_conv_weight_tensor) >= -np.max(first_scale_tensor.detach().numpy())
    assert np.max(second_conv_weight_tensor) <= np.max(second_scale_tensor.detach().numpy())
    assert np.min(second_conv_weight_tensor) >= -np.max(second_scale_tensor.detach().numpy())
