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
from nncf.torch.prepare_for_inference import convert_to_fakequantizer
from nncf.torch.prepare_for_inference import prepare_for_inference
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
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
    """Check that model contains only 8bit FakeQuantize operators."""

    if hasattr(model, "external_quantizers"):
        for key in list(model.external_quantizers.keys()):
            op = model.external_quantizers[key]
            assert isinstance(model.external_quantizers[key], FakeQuantize)
            assert op.quant_max - op.quant_min == 255

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key).op
                assert isinstance(op, FakeQuantize)
                assert op.quant_max - op.quant_min == 255

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.post_ops(key).op
                assert isinstance(op, FakeQuantize)
                assert op.quant_max - op.quant_min == 255


@pytest.mark.parametrize("mode", ("asymmetric", "symmetric"))
@pytest.mark.parametrize("overflow_fix", ("disable", "enable"))
def test_prepare_for_inference_quantization(mode, overflow_fix):
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init(model.INPUT_SIZE[-1])
    config["compression"]["weights"] = {"mode": mode}
    config["compression"]["activations"] = {"mode": mode}
    config["compression"]["overflow_fix"] = overflow_fix
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
                {
                    "algorithm": "quantization",
                    "initializer": {"range": {"num_init_samples": 0}},
                    "preset": "mixed",
                    "overflow_fix": "disable",
                },
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



@pytest.mark.parametrize("signed", (True, False))
@pytest.mark.parametrize("half_range", (True, False))
@pytest.mark.parametrize("scale", ([1.0], [0.3], [[[[0.3]]], [[[1.]]]]))
def test_converting_symetric(signed, half_range, scale):
    input_tensor = _gen_input_tensor([2, 1, 2, 2])
    scale = torch.Tensor(scale)

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=signed,
        narrow_range=False,
        scale_shape=scale.shape,
        logarithm_scale=False,
        half_range=half_range,
        is_quantized_on_export=True,
    )
    quantizer = SymmetricQuantizer(qspec)
    quantizer.scale.data = scale
    x_nncf = quantizer(input_tensor)

    fq = convert_to_fakequantizer(quantizer)
    x_torch = fq(input_tensor)

    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"



@pytest.mark.parametrize("signed", (True, False))
@pytest.mark.parametrize("half_range", (True, False))
@pytest.mark.parametrize("input_low, input_range", (
    ([0.0], [1.0]),
    ([-0.3], [0.9]),
    ([[[[0.0]]], [[[0.0]]]],  [[[[0.3]]], [[[1.]]]])
    )
)
def test_converting_asymetric(signed, half_range, input_low, input_range):
    input_tensor = _gen_input_tensor([2, 1, 2, 2])
    input_low = torch.Tensor(input_low)
    input_range = torch.Tensor(input_range)

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=signed,
        narrow_range=False,
        scale_shape=input_low.shape,
        logarithm_scale=False,
        half_range=half_range,
        is_quantized_on_export=True,
    )
    quantizer = AsymmetricQuantizer(qspec)
    quantizer.input_low.data = input_low
    quantizer.input_range.data = input_range
    x_nncf = quantizer(input_tensor)

    fq = convert_to_fakequantizer(quantizer)
    x_torch = fq(input_tensor)

    assert torch.all(torch.isclose(x_torch, x_nncf)), f"{x_torch.view(-1)} != {x_nncf.view(-1)}"
