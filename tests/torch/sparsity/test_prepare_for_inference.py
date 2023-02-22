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

import pytest
import torch

from nncf.config import NNCFConfig
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoController
from tests.common.quantization.data_generators import generate_lazy_sweep_data
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_prepare_for_inference import check_quantizer_operators


def _get_config_for_algo(input_size, quantization=False):
    config = NNCFConfig()
    config.update({"model": "model", "input_info": {"sample_size": input_size}, "compression": []})
    config["compression"].append(
        {
            "algorithm": "const_sparsity",
        }
    )

    if quantization:
        config["compression"].append(
            {
                "algorithm": "quantization",
                "initializer": {"range": {"num_init_samples": 0}},
            }
        )
        register_bn_adaptation_init_args(config)

    return config


@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_prepare_for_inference_sparsity(enable_quantization):
    input_size = [1, 1, 8, 8]
    model = BasicConvTestModel()
    config = _get_config_for_algo(input_size, enable_quantization)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    sparsity_ctrl = compression_ctrl
    if enable_quantization:
        for ctrl in compression_ctrl.child_ctrls:
            if isinstance(ctrl, BaseSparsityAlgoController):
                sparsity_ctrl = ctrl

    compressed_model.conv.weight.data = torch.ones_like(compressed_model.conv.weight.data)
    assert sparsity_ctrl.sparsified_module_info[0].operand.binary_mask.shape[0] == 2
    sparsity_ctrl.sparsified_module_info[0].operand.binary_mask[0] = 0

    input_tensor = generate_lazy_sweep_data(input_size)
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.prepare_for_inference()
    x_torch = inference_model(input_tensor)

    check_quantizer_operators(inference_model)

    nonzero_inference_model = torch.count_nonzero(inference_model.conv.weight.data)
    nonzero_binary_mask = torch.count_nonzero(compressed_model.conv.pre_ops["0"].operand.binary_mask)
    assert nonzero_inference_model == 4
    assert nonzero_inference_model == nonzero_binary_mask

    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"


@pytest.mark.parametrize("make_model_copy", (True, False))
@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_make_model_copy(make_model_copy, enable_quantization):
    model = BasicConvTestModel()
    config = _get_config_for_algo(model.INPUT_SIZE, enable_quantization)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    inference_model = compression_ctrl.prepare_for_inference(make_model_copy=make_model_copy)

    if make_model_copy:
        assert id(inference_model) != id(compressed_model)
    else:
        assert id(inference_model) == id(compressed_model)

    assert id(compressed_model) == id(compression_ctrl.model)
    if enable_quantization:
        for ctrl in compression_ctrl.child_ctrls:
            assert id(compressed_model) == id(ctrl.model)
