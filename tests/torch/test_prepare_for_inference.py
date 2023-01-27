"""
 Copyright (c) 2023 Intel Corporation
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

import numpy as np
import pytest
import torch
from torch.quantization.fake_quantize import FakeQuantize

from nncf.common.quantization.structs import QuantizationMode
from nncf.config import NNCFConfig
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
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import get_pruning_baseline_config
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init


def _gen_input_tensor(shape):
    n = np.prod(list(shape))
    res = torch.Tensor(range(n)) / torch.Tensor([n - 1]) * torch.Tensor([2]) - torch.Tensor([1.0])
    res[n // 2] = 0.0
    return res.reshape(shape)


def _check_operators(model):
    """Check that model contains only FakeQuantize operators."""

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                assert isinstance(op.op, FakeQuantize)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.get_pre_op(key)
                assert isinstance(op.op, FakeQuantize)


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

    prepare_for_inference(compressed_model, compression_ctrl)
    x_torch = compressed_model(input_tensor)

    _check_operators(compressed_model)

    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"


def test_prepare_for_inference_pruning():
    input_size = [1, 1, 8, 8]
    model = BigPruningTestModel()
    config = get_pruning_baseline_config(input_size)
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = 0.5
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = _gen_input_tensor(input_size)
    x_nncf = compressed_model(input_tensor)

    prepare_for_inference(compressed_model, compression_ctrl)
    x_torch = compressed_model(input_tensor)

    _check_operators(compressed_model)

    assert torch.equal(x_nncf, x_torch), f"{x_nncf=} != {x_torch}"


def test_prepare_for_inference_quantization_and_pruning():
    input_size = [1, 1, 8, 8]
    model = BigPruningTestModel()
    config = NNCFConfig()
    config.update(
        {
            "model": "model",
            "input_info": {"sample_size": input_size},
            "compression": [
                {
                    "algorithm": "filter_pruning",
                    "params": {"schedule": "baseline", "num_init_steps": 1},
                    "pruning_init": 0.5,
                },
                {"algorithm": "quantization", "initializer": {"range": {"num_init_samples": 0}}, "preset": "mixed"},
            ],
        }
    )
    register_bn_adaptation_init_args(config)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = _gen_input_tensor(input_size)
    x_nncf = compressed_model(input_tensor)

    prepare_for_inference(compressed_model, compression_ctrl)
    x_torch = compressed_model(input_tensor)

    _check_operators(compressed_model)

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
        ([0.0], [0.9], 255, 0, 256, 0.1),
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

    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"


@pytest.mark.parametrize(
    "level_high, level_low, levels, scale, eps, zero_point",
    (
        (127, -128, 256, [[[[1.3, 1.0]]]], 1e-16, None),
        (255, 0, 256, [[[[1.3, 1.0]]]], 1e-16, None),
        (255, 0, 256, [[[[1.3, 1.0]]]], 1e-16, [[[[0.0, 0.0]]]]),
        (255, 0, 256, [1.5], 1e-16, None),
        (127, -128, 256, [1.5], 1e-16, None),
        (127, -128, 256, [1.5], 1e-16, [0.0]),
        (255, 0, 256, [1.5], 0, None),
        (127, -128, 256, [1.5], 0, None),
        (127, -128, 256, [1.5], 0.1, None),
    ),
)
def test_convert_symmetric_parameters(level_high, level_low, levels, scale, eps, zero_point):
    input_tensor = _gen_input_tensor([2, 1, 2, 2])

    scale = torch.Tensor(scale)
    if zero_point is not None:
        zero_point = torch.Tensor(zero_point)
    scale_shape = scale.shape
    ch_axis = np.argmax(scale_shape)

    x_nncf = symmetric_quantize(input_tensor, levels, level_low, level_high, scale, eps, skip=False)
    quant_max, quant_min, scale, zero_point = convert_symmetric_parameters(
        level_high, level_low, scale, eps, zero_point
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

    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"
