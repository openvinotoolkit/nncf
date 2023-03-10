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
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.prepare_for_inference import convert_to_torch_fakequantizer
from tests.common.quantization.data_generators import check_outputs
from tests.common.quantization.data_generators import generate_lazy_sweep_data
from tests.common.quantization.data_generators import generate_random_low_and_range_by_input_size
from tests.common.quantization.data_generators import generate_random_scale_by_input_size
from tests.common.quantization.data_generators import generate_sweep_data
from tests.common.quantization.data_generators import get_quant_len_by_range
from tests.common.quantization.data_generators import get_symmetric_range_level
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_functions import get_test_data


def _get_config_for_algo(input_size, quant_mode="symmetric", overflow_fix="enable", bits=8):
    config = NNCFConfig()
    config.update({"model": "model", "input_info": {"sample_size": input_size}, "compression": []})
    config["target_device"] = "TRIAL"
    config["compression"].append(
        {
            "algorithm": "quantization",
            "initializer": {"range": {"num_init_samples": 0}},
            "weights": {"mode": quant_mode, "bits": bits},
            "activations": {"mode": quant_mode, "bits": bits},
            "overflow_fix": overflow_fix,
        }
    )

    return config


def _idfn(val):
    if isinstance(val, tuple):
        return "{}".format("-".join([str(v) for v in val]))
    return None


def check_quantizer_operators(model, levels=255):
    """Check that model contains only 8bit FakeQuantize operators."""

    if hasattr(model, "external_quantizers"):
        for key in list(model.external_quantizers.keys()):
            op = model.external_quantizers[key]
            assert isinstance(model.external_quantizers[key], FakeQuantize)
            assert op.quant_max - op.quant_min == levels

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key).op
                assert isinstance(op, FakeQuantize)
                assert op.quant_max - op.quant_min == levels

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.post_ops(key).op
                assert isinstance(op, FakeQuantize)
                assert op.quant_max - op.quant_min == levels


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
@pytest.mark.parametrize("num_bits", (4, 8), ids=("4-bits", "8-bits"))
def test_converting_symmetric_quantizer(
    input_size, num_bits, is_per_channel, is_weights, half_range, is_signed, use_cuda
):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    if is_per_channel and input_size[0 if is_weights else 1] == 1:
        pytest.skip("Same case as for per_tensor case")

    np.random.seed(42)
    real_num_bits = num_bits - 1 if half_range else num_bits
    np_scale = generate_random_scale_by_input_size(input_size, is_per_channel, is_weights)
    tensor_scale = get_test_data([np_scale], use_cuda)

    level_low, level_high, _ = get_symmetric_range_level(is_signed, real_num_bits)

    input_low = np_scale * (level_low / level_high)
    input_range = np_scale - input_low

    input_low = np.squeeze(input_low)
    input_range = np.squeeze(input_range)

    qspec = PTQuantizerSpec(
        num_bits=num_bits,
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

    tuned_input_low, tuned_input_high = quantizer.get_input_low_input_high()
    if use_cuda:
        tuned_input_low = tuned_input_low.cpu()
        tuned_input_high = tuned_input_high.cpu()
    tuned_input_low = tuned_input_low.detach().numpy().squeeze()
    tuned_input_high = tuned_input_high.detach().numpy().squeeze()
    tuned_input_range = tuned_input_high - tuned_input_low

    np_input, np_is_near_mid_point, quant_lens = generate_sweep_data(
        input_size, tuned_input_low, tuned_input_range, real_num_bits, is_per_channel, is_weights
    )
    test_input = get_test_data([np_input], use_cuda)

    x_nncf = quantizer(test_input)

    fq = convert_to_torch_fakequantizer(quantizer)

    fq_levels = fq.quant_max - fq.quant_min
    assert fq_levels == 2**num_bits - 1, "Levels in converted FQ should be 2**num_bits-1"

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
@pytest.mark.parametrize("num_bits", (4, 8), ids=("4-bits", "8-bits"))
def test_converting_asymmetric_quantizer(input_size, num_bits, is_per_channel, is_weights, half_range, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    if is_per_channel and input_size[0 if is_weights else 1] == 1:
        pytest.skip("Same case as for per_tensor case")

    np.random.seed(42)
    real_num_bits = num_bits - 1 if half_range else num_bits

    input_low, input_range = generate_random_low_and_range_by_input_size(input_size, is_per_channel, is_weights)

    ######################################################################
    # TODO: Workaround for issue 105241 (remove after fix)
    get_quant_len = get_quant_len_by_range(input_range, real_num_bits)
    input_low[(input_low > -get_quant_len / 2) & (input_low < 0)] = 0
    ######################################################################

    tensor_input_low, tensor_input_range = get_test_data([input_low, input_range], use_cuda)

    input_low = np.squeeze(input_low)
    input_range = np.squeeze(input_range)

    qspec = PTQuantizerSpec(
        num_bits=num_bits,
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

    tuned_input_low, tuned_input_high = quantizer.get_input_low_input_high()
    if use_cuda:
        tuned_input_low = tuned_input_low.cpu()
        tuned_input_high = tuned_input_high.cpu()
    tuned_input_low = tuned_input_low.detach().numpy().squeeze()
    tuned_input_high = tuned_input_high.detach().numpy().squeeze()
    tuned_input_range = tuned_input_high - tuned_input_low

    np_input, np_is_near_mid_point, quant_lens = generate_sweep_data(
        input_size, tuned_input_low, tuned_input_range, real_num_bits, is_per_channel, is_weights
    )
    test_input = get_test_data([np_input], use_cuda)

    x_nncf = quantizer(test_input)

    fq = convert_to_torch_fakequantizer(quantizer)

    fq_levels = fq.quant_max - fq.quant_min
    assert fq_levels == 2**num_bits - 1, "Levels in converted FQ should be 2**num_bits-1"

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


@pytest.mark.parametrize("mode", ("asymmetric", "symmetric"))
@pytest.mark.parametrize("overflow_fix", ("disable", "enable"), ids=("overflow_fix_enable", "overflow_fix_disable"))
@pytest.mark.parametrize("num_bits", (4, 8), ids=("4-bits", "8-bits"))
def test_prepare_for_inference_quantization(mode, overflow_fix, num_bits):
    model = BasicConvTestModel()

    config = _get_config_for_algo(model.INPUT_SIZE, mode, overflow_fix, bits=num_bits)
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = generate_lazy_sweep_data(model.INPUT_SIZE)
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.prepare_for_inference()
    x_torch = inference_model(input_tensor)

    check_quantizer_operators(inference_model, 2**num_bits - 1)

    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"


@pytest.mark.parametrize("make_model_copy", (True, False))
def test_make_model_copy(make_model_copy):
    model = BasicConvTestModel()
    config = _get_config_for_algo(model.INPUT_SIZE)
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    inference_model = compression_ctrl.prepare_for_inference(make_model_copy=make_model_copy)

    if make_model_copy:
        assert id(inference_model) != id(compressed_model)
    else:
        assert id(inference_model) == id(compressed_model)

    assert id(compressed_model) == id(compression_ctrl.model)
