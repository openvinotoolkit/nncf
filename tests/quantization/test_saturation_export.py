import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
from nncf import NNCFConfig
from nncf.layers import NNCFConv2d
from tests.helpers import create_conv, create_compressed_model_and_algo_for_test
from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer


def test_is_correct_saturation_fix_levels():
    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert (quantizer.is_saturation_fix == True)
    assert (quantizer.levels == 128)
    assert (quantizer.level_low == -64)
    assert (quantizer.level_high == 63)

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = AsymmetricQuantizer(q_config)

    assert (quantizer.is_saturation_fix == True)
    assert (quantizer.levels == 128)
    assert (quantizer.level_low == 0)
    assert (quantizer.level_high == 127)

    q_config = QuantizerConfig(
        bits=7,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=True,
        is_weights=True,
        input_shape=[3, 32, 32],
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert (quantizer.is_saturation_fix == True)
    assert (quantizer.levels == 64)
    assert (quantizer.level_low == -32)
    assert (quantizer.level_high == 31)

    q_config = QuantizerConfig(
        bits=4,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert (quantizer.is_saturation_fix == True)
    assert (quantizer.levels == 8)
    assert (quantizer.level_low == -4)
    assert (quantizer.level_high == 3)

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=False,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)

    assert (quantizer.is_saturation_fix == True)
    assert (quantizer.levels == 128)
    assert (quantizer.level_low == -64)
    assert (quantizer.level_high == 63)


def get_config_for_export_mode(should_be_onnx_standard: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update({
        "input_info": {
            "sample_size": [1, 1, 20, 20]
        },
        "compression": {
            "algorithm": "quantization",
            "export_to_onnx_standard_ops": should_be_onnx_standard
        },
        "target_device": "CPU"
    })
    return nncf_config


class TwoConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []
        self.features.append(create_conv(1, 2, 2, -1, -2))
        self.features.append(create_conv(2, 1, 3, 0, 0))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)


def test_hw_config_saturation_issue_set():
    # Test CPU device in which we use saturation issue
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(True)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for conv in [module for module in compressed_model.get_nncf_wrapped_model().features if
                 isinstance(module, NNCFConv2d)]:
        assert (conv.pre_ops._modules['0'].op.is_saturation_fix == True)
        assert (conv.pre_ops._modules['0'].op.levels == 128)
        assert (conv.pre_ops._modules['0'].op.level_low == -64)
        assert (conv.pre_ops._modules['0'].op.level_high == 63)

    for quantizer in compressed_model.activation_quantizers.values():
        assert (quantizer.is_saturation_fix == False)

    # Test ANY target device in which we use saturation issue
    model = TwoConvTestModel()
    nncf_config.update({"target_device": "ANY"})

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    for conv in [module for module in compressed_model.get_nncf_wrapped_model().features if
                 isinstance(module, NNCFConv2d)]:
        assert (conv.pre_ops._modules['0'].op.is_saturation_fix == True)
        assert (conv.pre_ops._modules['0'].op.levels == 128)
        assert (conv.pre_ops._modules['0'].op.level_low == -64)
        assert (conv.pre_ops._modules['0'].op.level_high == 63)

    for quantizer in compressed_model.activation_quantizers.values():
        assert (quantizer.is_saturation_fix == False)

    # Test other devices in which we don't use saturation issue
    def test_helper(target_device):
        model = TwoConvTestModel()
        nncf_config.update({"target_device": target_device})

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

        for conv in [module for module in compressed_model.get_nncf_wrapped_model().features if
                     isinstance(module, NNCFConv2d)]:
            assert (conv.pre_ops._modules['0'].op.is_saturation_fix == False)
        for quantizer in compressed_model.activation_quantizers.values():
            assert (quantizer.is_saturation_fix == False)

    for device in ['VPU', 'GPU', 'TRIAL']:
       test_helper(device)



def test_onnx_export_weights_tensors_clipped(tmp_path):
    model = TwoConvTestModel()
    nncf_config = get_config_for_export_mode(True)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # Set scale tensors
    first_scale_tensor = torch.tensor((1, 1), dtype=torch.float32, requires_grad=True)
    second_scale_tensor = torch.tensor(3000, dtype=torch.float32, requires_grad=True)

    # Update scale tensors in the model
    compressed_model.get_nncf_wrapped_model().features[0].pre_ops._modules['0'].op.scale.data = torch.nn.Parameter(
        first_scale_tensor)
    compressed_model.get_nncf_wrapped_model().features[1].pre_ops._modules['0'].op.scale.data = torch.nn.Parameter(
        second_scale_tensor)

    # Set weight tensors bigger then scale tensor
    compressed_model.get_nncf_wrapped_model().features[0].weight.data = 1000 * torch.ones_like(
        compressed_model.get_nncf_wrapped_model().features[0].weight)

    # Set weight tensors in range of scale tensor
    compressed_model.get_nncf_wrapped_model().features[1].weight.data = 2000 * torch.ones_like(
        compressed_model.get_nncf_wrapped_model().features[1].weight)

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

    assert (np.max(first_conv_weight_tensor) <= np.max(first_scale_tensor.detach().numpy()))
    assert (np.min(first_conv_weight_tensor) >= -np.max(first_scale_tensor.detach().numpy()))
    assert (np.max(second_conv_weight_tensor) <= np.max(second_scale_tensor.detach().numpy()))
    assert (np.min(second_conv_weight_tensor) >= -np.max(second_scale_tensor.detach().numpy()))
