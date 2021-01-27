import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
from onnx import numpy_helper

from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test, create_conv
from tests.quantization.test_onnx_export import get_config_for_export_mode


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

    # Test CPU, ANY devices in which we use saturation issue
    def test_with_saturation_helper(target_device):
        model = TwoConvTestModel()
        nncf_config.update({"target_device": target_device})

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

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

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

        for quantizer in compression_ctrl.weight_quantizers.values():
            assert not quantizer.quantizer_module_ref.is_saturation_fix
        for quantizer in compression_ctrl.non_weight_quantizers.values():
            assert not quantizer.quantizer_module_ref.is_saturation_fix

    for device in ['CPU', 'ANY']:
        test_with_saturation_helper(device)

    for device in ['VPU', 'GPU', 'TRIAL']:
        test_without_saturation_helper(device)


class EightConvTestModel(nn.Module):
    def __init__(self, in_out_ch):
        super().__init__()
        self.features = []
        self.features.append(nn.Sequential(create_conv(*in_out_ch[0], 2, -1, -2)))
        self.features.append(nn.Sequential(create_conv(*in_out_ch[1], 5, 1, 1)))
        self.features.append(nn.Sequential(create_conv(*in_out_ch[2], 1, 2, 2)))
        self.features.append(nn.Sequential(create_conv(*in_out_ch[3], 9, -1, 0)))
        self.features.append(nn.Sequential(create_conv(*reversed(in_out_ch[3]), 3, 0, 1)))
        self.features.append(nn.Sequential(create_conv(*reversed(in_out_ch[2]), 1, -1, 9)))
        self.features.append(nn.Sequential(create_conv(*reversed(in_out_ch[1]), 2, 10, 1)))
        self.features.append(nn.Sequential(create_conv(*reversed(in_out_ch[0]), 1, 1, 1)))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)


def test_are_fq_exported_per_channel_weights_tensors_clipped(tmp_path):
    in_out_ch = [[1, 3], [3, 5], [5, 7], [7, 10]]
    model = EightConvTestModel(in_out_ch)
    nncf_config = get_config_for_export_mode(False)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # Update scale tensors in the model
    quantizers = compression_ctrl.weight_quantizers.values()
    for quantizer in quantizers:
        quantizer.quantizer_module_ref.scale = torch.nn.Parameter(
            torch.rand(quantizer.quantized_module.weight.shape[0], dtype=torch.float32, requires_grad=True))

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    onnx_conv_weight_tensors = []

    # Add conv weights tensors
    # pylint:disable=no-member
    for weight_tensor in onnx_model.graph.initializer:
        if 'weight' in weight_tensor.name:
            onnx_conv_weight_tensors.append(numpy_helper.to_array(weight_tensor))

    # Add last convolution weights
    for node in onnx_model.graph.node:
        try:
            if node.op_type == 'Constant':
                if node.attribute[0].t.dims[0] == 1 and node.attribute[0].t.dims[1] == 3 and \
                        node.attribute[0].t.dims[2] == 1 and node.attribute[0].t.dims[3] == 1:
                    onnx_conv_weight_tensors.append(numpy_helper.to_array(node.attribute[0].t))
        except:
            continue

    level_positive_negative_ratio = 64. / 63.

    for quantizer, filter in zip(quantizers, onnx_conv_weight_tensors):
        for scale, channel in zip(quantizer.quantizer_module_ref.scale, filter):
            negative_max_val = level_positive_negative_ratio * scale
            assert scale >= np.max(channel)
            assert -negative_max_val <= np.min(channel)


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

    # Set weight tensors bigger then scale tensor
    first_quantizer.quantized_module.weight.data = -0.5 + torch.ones_like(
        first_quantizer.quantized_module.weight)
    second_quantizer.quantized_module.weight.data = 2000 * torch.ones_like(
        second_quantizer.quantized_module.weight)

    onnx_checkpoint_path = str('/home/aleksei/model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    onnx_model = onnx.load(onnx_checkpoint_path)

    # Find weight tensors in ONNX model
    onnx_conv_weight_tensors = []

    # Add conv weights tensors
    # pylint:disable=no-member
    for weight_tensor in onnx_model.graph.initializer:
        if 'weight' in weight_tensor.name:
            onnx_conv_weight_tensors.append(numpy_helper.to_array(weight_tensor))

    # pylint:disable=no-member
    for node in onnx_model.graph.node:
        try:
            if node.op_type == 'Constant':
                if node.attribute[0].t.dims[0] == 1 and node.attribute[0].t.dims[1] == 2 and \
                        node.attribute[0].t.dims[2] == 3 and node.attribute[0].t.dims[3] == 3:
                    onnx_conv_weight_tensors.append(numpy_helper.to_array(node.attribute[0].t))
        except (AttributeError, IndexError):
            continue

    level_positive_negative_ratio = 64. / 63.

    assert np.max(onnx_conv_weight_tensors[0]) <= np.max(first_scale_tensor.detach().numpy())
    assert np.min(onnx_conv_weight_tensors[0]) >= level_positive_negative_ratio * -np.max(
        first_scale_tensor.detach().numpy())
    assert np.max(onnx_conv_weight_tensors[1]) <= np.max(second_scale_tensor.detach().numpy())
    assert np.min(onnx_conv_weight_tensors[1]) >= level_positive_negative_ratio * -np.max(
        second_scale_tensor.detach().numpy())


def test_is_pytorch_output_the_same_as_onnx_qdq_saturation_fix_applied(tmp_path):
    model = TwoConvTestModel()

    nncf_config = get_config_for_export_mode(True)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path)
    input_tensors = [np.random.normal(size=[1, 1, 4, 4]), np.random.uniform(size=[1, 1, 4, 4]),
                     100 * np.random.normal(size=[1, 1, 4, 4]), 100 * np.random.uniform(size=[1, 1, 4, 4])]
    for input_tensor in input_tensors:
        torch_input = torch.tensor(input_tensor, dtype=torch.float32)

        with torch.no_grad():
            torch_out = compressed_model.forward(torch_input)

        # ONNXRuntime
        sess = rt.InferenceSession(onnx_checkpoint_path)
        input_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {input_name: input_tensor.astype(np.float32)})[0]

        assert (np.allclose(torch_out.numpy(), onnx_out))
