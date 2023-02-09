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
from tests.common.quantization.data_generators import check_outputs
from tests.common.quantization.data_generators import generate_random_low_and_range_by_input_size
from tests.common.quantization.data_generators import generate_scale_by_input_size
from tests.common.quantization.data_generators import generate_test_input
from tests.common.quantization.data_generators import get_symmetric_range_level
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import get_pruning_baseline_config
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.test_functions import get_test_data


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


def _idfn(val):
    if isinstance(val, tuple):
        return "{}".format("-".join([str(v) for v in val]))
    return None


INPUT_TEST_SCALES = (
    (1, 16, 24, 24),
    (16, 8, 12, 12),
)


@pytest.mark.parametrize("use_cuda", (True, False), ids=("cuda", "cpu"))
@pytest.mark.parametrize("half_range", [False, True], ids=["full_range", "half_range"])
@pytest.mark.parametrize("is_weights", (True, False), ids=("weights", "activation"))
@pytest.mark.parametrize("is_signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("input_size", INPUT_TEST_SCALES, ids=_idfn)
@pytest.mark.parametrize("is_per_channel", (True, False), ids=("per_channel", "per_tensor"))
def test_converting_symmetric_quantizer(input_size, is_per_channel, is_weights, half_range, is_signed, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    if is_per_channel and input_size[0 if is_weights else 1] == 1:
        pytest.skip("Same case as for per_tensor case")

    # np.random.seed(42)
    bits = 7 if half_range else 8
    np_scale = generate_scale_by_input_size(input_size, is_per_channel, is_weights)
    tensor_scale = get_test_data([np_scale], use_cuda)

    level_low, level_high, _ = get_symmetric_range_level(is_signed, bits)

    input_low = np_scale * (level_low / level_high)
    input_range = np_scale - input_low

    input_low = np.squeeze(input_low)
    input_range = np.squeeze(input_range)

    np_input, np_is_near_mid_point, quant_lens = generate_test_input(
        input_size, input_low, input_range, bits, is_per_channel, is_weights
    )
    test_input = get_test_data([np_input], use_cuda)

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=is_signed,
        narrow_range=False,
        scale_shape=tuple(tensor_scale.shape),
        logarithm_scale=False,
        half_range=half_range,
        is_quantized_on_export=True,
    )
    quantizer = SymmetricQuantizer(qspec)
    quantizer.scale.data = tensor_scale

    x_nncf = quantizer(test_input)

    fq = convert_to_fakequantizer(quantizer)

    fq_levels = fq.quant_max - fq.quant_min
    assert fq_levels == 255, "Levels in converted FQ should be 255"

    fq_test_input = test_input
    if half_range:
        # Required clamp of input for half range
        fq_input_low, fq_input_high = quantizer.get_input_low_input_high()
        fq_test_input = torch.min(torch.max(fq_test_input, fq_input_low), fq_input_high)
    x_torch = fq(fq_test_input)

    if use_cuda:
        test_input = test_input.cpu()
        x_nncf = x_nncf.cpu()
        x_torch = x_torch.cpu()

    check_outputs(x_nncf.detach().numpy(), x_torch.detach().numpy(), np_is_near_mid_point, quant_lens)


@pytest.mark.parametrize("use_cuda", (True, False), ids=("cuda", "cpu"))
@pytest.mark.parametrize("half_range", [False, True], ids=["full_range", "half_range"])
@pytest.mark.parametrize("is_weights", (True, False), ids=("weights", "activation"))
@pytest.mark.parametrize("input_size", INPUT_TEST_SCALES, ids=_idfn)
@pytest.mark.parametrize("is_per_channel", (True, False), ids=("per_channel", "per_tensor"))
def test_converting_asymmetric_quantizer(input_size, is_per_channel, is_weights, half_range, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    if is_per_channel and input_size[0 if is_weights else 1] == 1:
        pytest.skip("Same case as for per_tensor case")

    np.random.seed(42)
    bits = 7 if half_range else 8

    input_low, input_range = generate_random_low_and_range_by_input_size(input_size, is_per_channel, is_weights)
    tensor_input_low, tensor_input_range = get_test_data([input_low, input_range], use_cuda)

    input_low = np.squeeze(input_low)
    input_range = np.squeeze(input_range)

    np_input, np_is_near_mid_point, quant_lens = generate_test_input(
        input_size, input_low, input_range, bits, is_per_channel, is_weights
    )
    test_input = get_test_data([np_input], use_cuda)

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=False,
        narrow_range=False,
        scale_shape=tensor_input_low.shape,
        logarithm_scale=False,
        half_range=half_range,
        is_quantized_on_export=True,
    )
    quantizer = AsymmetricQuantizer(qspec)
    quantizer.input_low.data = tensor_input_low
    quantizer.input_range.data = tensor_input_range
    x_nncf = quantizer(test_input)

    fq = convert_to_fakequantizer(quantizer)

    fq_levels = fq.quant_max - fq.quant_min
    assert fq_levels == 255, "Levels in converted FQ should be 255"

    fq_test_input = test_input
    if half_range:
        # Required clamp of input for half range
        input_low, input_high = quantizer.get_input_low_input_high()
        fq_test_input = torch.min(torch.max(fq_test_input, input_low), input_high)

    x_torch = fq(fq_test_input)

    if use_cuda:
        test_input = test_input.cpu()
        x_nncf = x_nncf.cpu()
        x_torch = x_torch.cpu()

    check_outputs(x_nncf.detach().numpy(), x_torch.detach().numpy(), np_is_near_mid_point, quant_lens)

@pytest.mark.parametrize("mode", ("asymmetric", "symmetric"))
@pytest.mark.parametrize("overflow_fix", ("disable", "enable"), ids=("overflow_fix_enable", "overflow_fix_disable"))
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


@pytest.mark.parametrize("save_original_model", (True, False))
def test_save_original_model(save_original_model):
    model = BasicConvTestModel()
    config = get_quantization_config_without_range_init(model.INPUT_SIZE[-1])
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    modified_model = prepare_for_inference(compressed_model, compression_ctrl, save_original_model)

    if save_original_model:
        assert id(modified_model) != id(compressed_model)
    else:
        assert id(modified_model) == id(compressed_model)

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

    conv2_wight = compressed_model.nncf_module.conv2.weight.data
    assert torch.count_nonzero(conv2_wight) * 2 == torch.numel(conv2_wight), "Model was not pruned"

    assert torch.equal(x_nncf, x_torch), f"{x_nncf=} != {x_torch}"


@pytest.mark.parametrize("mode", ("asymmetric", "symmetric"))
@pytest.mark.parametrize("overflow_fix", ("disable", "enable"), ids=("overflow_fix_disable", "overflow_fix_enable"))
def test_prepare_for_inference_quantization_and_pruning(mode, overflow_fix):
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
                    "weights": {"mode": mode},
                    "activations": {"mode": mode},
                    "overflow_fix": overflow_fix,
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

    conv2_wight = compressed_model.nncf_module.conv2.weight.data
    assert torch.count_nonzero(conv2_wight) * 2 == torch.numel(conv2_wight), "Model was not pruned"

    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"
