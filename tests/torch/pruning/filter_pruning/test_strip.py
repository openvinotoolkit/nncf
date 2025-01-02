# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from nncf.config import NNCFConfig
from tests.common.quantization.data_generators import generate_lazy_sweep_data
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.quantization.test_strip import check_quantizer_operators


def _get_config_for_algo(input_size, quantization=False):
    config = NNCFConfig()
    config.update({"model": "model", "input_info": {"sample_size": input_size}, "compression": []})
    config["compression"].append(
        {
            "algorithm": "filter_pruning",
            "params": {"schedule": "baseline", "num_init_steps": 1},
            "pruning_init": 0.5,
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
def test_strip_pruning(enable_quantization):
    input_size = [1, 1, 8, 8]
    model = BigPruningTestModel().eval()
    config = _get_config_for_algo(input_size, enable_quantization)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = torch.Tensor(generate_lazy_sweep_data(input_size))
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.strip()
    x_torch = inference_model(input_tensor)

    check_quantizer_operators(inference_model)

    conv2_weight = inference_model.conv2.weight.data
    assert torch.count_nonzero(conv2_weight) * 2 == torch.numel(conv2_weight), "Model was not pruned"

    assert torch.equal(x_nncf, x_torch), f"{x_nncf=} != {x_torch}"


@pytest.mark.parametrize("do_copy", (True, False))
@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_do_copy(do_copy, enable_quantization):
    model = BasicConvTestModel()
    config = _get_config_for_algo(model.INPUT_SIZE, enable_quantization)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    inference_model = compression_ctrl.strip(do_copy=do_copy)

    if do_copy:
        assert id(inference_model) != id(compressed_model)
    else:
        assert id(inference_model) == id(compressed_model)

    assert id(compressed_model) == id(compression_ctrl.model)
    if enable_quantization:
        for ctrl in compression_ctrl.child_ctrls:
            assert id(compressed_model) == id(ctrl.model)
