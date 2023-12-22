# Copyright (c) 2023 Intel Corporation
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

from nncf import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization import compress_weights

DATA_BASED_SENSITIVITY_METRICS = (
    SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
    SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
)

ALL_SENSITIVITY_METRICS = DATA_BASED_SENSITIVITY_METRICS + (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,)

SUPPORTED_MODES = (CompressWeightsMode.INT8, CompressWeightsMode.INT8_ASYM)
UNSUPPORTED_MODES = (
    CompressWeightsMode.INT4_SYM,
    CompressWeightsMode.INT4_ASYM,
    CompressWeightsMode.NF4,
    CompressWeightsMode.INT8_SYM,
)


class ShortTransformer(torch.nn.Module):
    def __init__(self, in_features, num_embeddings, share_weights=False):
        super().__init__()
        self.wte = torch.nn.Embedding(num_embeddings, in_features)
        self.linear = torch.nn.Linear(in_features, in_features)
        self.lm_head = torch.nn.Linear(in_features, num_embeddings)

        if share_weights:
            self.lm_head.weight = self.wte.weight

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = self.linear(x)
        res = self.lm_head(x)
        return res


def test_compress_weights():
    model = ShortTransformer(5, 10)

    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype in [torch.uint8, torch.int8]:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules


def test_compress_shared_weights():
    model = ShortTransformer(5, 10, share_weights=True)

    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype in [torch.uint8, torch.int8]:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules

    assert len(compressed_model.wte.pre_ops) > 0

    assert len(compressed_model.wte.pre_ops) == len(compressed_model.lm_head.pre_ops)

    for key, val in compressed_model.wte.pre_ops.items():
        assert compressed_model.lm_head.get_pre_op(key) is val


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
@pytest.mark.parametrize(
    "params",
    (
        {"ratio": 0.5},
        {"group_size": 64},
        {"all_layers": True},
        {"all_layers": False},
        *({"sensitivity_metric": metric} for metric in ALL_SENSITIVITY_METRICS),
        {"dataset": "anything"},
        {"ignored_scope": "anything"},
    ),
)
def test_raise_error_with_unsupported_params_for_int8(mode, params):
    dummy_torch_model = torch.nn.Module()
    with pytest.raises(AttributeError):
        compress_weights(dummy_torch_model, mode=mode, **params)


@pytest.mark.parametrize("mode", UNSUPPORTED_MODES)
def test_raise_error_with_not_int8_asym(mode):
    dummy_torch_model = torch.nn.Module()
    with pytest.raises(AttributeError):
        compress_weights(dummy_torch_model, mode=mode)
