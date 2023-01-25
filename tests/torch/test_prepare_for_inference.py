import numpy as np
import pytest
import torch

from nncf.common.quantization.structs import QuantizationMode
from nncf.torch.prepare_for_inference import convert_asymmetric_parameters
from nncf.torch.prepare_for_inference import convert_symmetric_parameters
from nncf.torch.prepare_for_inference import convert_to_fakequantizer
from nncf.torch.prepare_for_inference import prepare_for_inference
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import asymmetric_quantize
from nncf.torch.quantization.quantize_functions import symmetric_quantize
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init


def _gen_input_tensor(shape):
    n = np.prod(list(shape))
    res = torch.Tensor(range(n)) / torch.Tensor([n - 1]) * torch.Tensor([2]) - torch.Tensor([1.0])
    res[n // 2] = 0.0
    return res.reshape(shape)


@pytest.mark.parametrize("mode", ("asymmetric", "symmetric"))
def test_prepare_for_inference_quantization(mode):
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init(model.INPUT_SIZE[-1])
    config["compression"]["weights"] = {"mode": mode}
    config["compression"]["activations"] = {"mode": mode}
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = _gen_input_tensor(model.INPUT_SIZE)
    x_nncf = compressed_model(input_tensor)

    inference_model = prepare_for_inference(compressed_model, compression_ctrl)
    x_torch = inference_model(input_tensor)

    # assert torch.equal(x_nncf, x_torch), f"{x_nncf=} != {x_torch}"
    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"


@pytest.mark.parametrize(
    "quantizer_mode, input_shape, scale_shape",
    (
        (QuantizationMode.ASYMMETRIC, (2, 1, 2, 2), (2, 1, 1, 1)),
        (QuantizationMode.ASYMMETRIC, (1, 1, 4, 4), (1,)),
        (QuantizationMode.SYMMETRIC, (2, 1, 2, 2), (2, 1, 1, 1)),
        (QuantizationMode.SYMMETRIC, (1, 1, 4, 4), (1,)),
    ),
)
def test_convert_to_fakequantizer(quantizer_mode, input_shape, scale_shape):
    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=quantizer_mode,
        signedness_to_force=True,
        narrow_range=False,
        scale_shape=scale_shape,
        logarithm_scale=False,
        half_range=False,
        is_quantized_on_export=True,
    )

    quantizer = (
        SymmetricQuantizer(qspec) if quantizer_mode == QuantizationMode.SYMMETRIC else AsymmetricQuantizer(qspec)
    )
    quantizer.train(False)

    input_tensor = _gen_input_tensor(input_shape)
    x_nncf = quantizer(input_tensor).data

    converted_fq = convert_to_fakequantizer(quantizer)
    x_torch = converted_fq(input_tensor)

    # assert torch.equal(x_nncf, x_torch), f"{x_nncf=} != {x_torch}"
    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"


@pytest.mark.parametrize(
    "input_low, input_range, level_high, level_low, levels, eps",
    (
        ([0.0], [1.0], 127, -128, 256, 1e-16),
        ([0.0], [1.0], 255, 0, 256, 1e-16),
        ([0.0], [1.0], 127, -128, 256, 0),
        ([0.0], [1.0], 255, 0, 256, 0),
        ([0.0], [0.5], 127, -128, 256, 1e-16),
        ([0.5], [0.6], 255, 0, 256, 1e-16),
        ([[[[0.0]]], [[[0.0]]]], [[[[1.0]]], [[[1.0]]]], 127, 0, 128, 1e-16),
        ([[[[0.0]]], [[[0.0]]]], [[[[1.0]]], [[[1.0]]]], 127, 0, 128, 0),
    ),
)
def test_convert_asymmetric_parameters(input_low, input_range, level_high, level_low, levels, eps):
    input_tensor = _gen_input_tensor([2, 2, 2, 2])

    input_low = torch.Tensor(input_low)
    input_range = torch.Tensor(input_range)
    scale_shape = input_low.shape
    ch_axis = np.argmax(scale_shape)

    x_nncf = asymmetric_quantize(input_tensor, levels, level_low, level_high, input_low, input_range, eps, False)

    quant_max, quant_min, scale, zero_point = convert_asymmetric_parameters(
        level_high, level_low, input_low, input_range, levels, eps
    )

    if len(scale_shape) == 1:
        x_torch = torch.fake_quantize_per_tensor_affine(
            input_tensor,
            scale,
            zero_point,
            quant_min,
            quant_max,
        )
    else:
        x_torch = torch.fake_quantize_per_channel_affine(
            input_tensor,
            scale,
            zero_point,
            ch_axis,
            quant_min,
            quant_max,
        )

    # assert torch.equal(x_torch, x_nncf), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"
    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"


@pytest.mark.parametrize(
    "level_high, level_low, levels, scale, eps",
    (
        (127, -128, 256, [[[[1.3, 1.0]]]], 1e-16),
        (255, 0, 256, [[[[1.3, 1.0]]]], 1e-16),
        (255, 0, 256, [1.5], 1e-16),
        (127, -128, 256, [1.5], 1e-16),
        (255, 0, 256, [1.5], 0),
        (127, -128, 256, [1.5], 0),
    ),
)
def test_convert_symmetric_parameters(level_high, level_low, levels, scale, eps):
    input_tensor = _gen_input_tensor([2, 1, 2, 2])

    scale = torch.Tensor(scale)
    scale_shape = scale.shape
    ch_axis = np.argmax(scale_shape)

    x_nncf = symmetric_quantize(input_tensor, levels, level_low, level_high, scale, eps, skip=False)
    quant_max, quant_min, scale, zero_point = convert_symmetric_parameters(level_high, level_low, scale, eps, None)

    if len(scale_shape) == 1:
        x_torch = torch.fake_quantize_per_tensor_affine(
            input_tensor,
            scale,
            zero_point,
            quant_min,
            quant_max,
        )
    else:
        x_torch = torch.fake_quantize_per_channel_affine(
            input_tensor,
            scale,
            zero_point,
            ch_axis,
            quant_min,
            quant_max,
        )

    # assert torch.equal(x_torch, x_nncf), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"
    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"


# TODO: tests with prunning
