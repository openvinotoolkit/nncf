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
from typing import Optional
from unittest.mock import Mock
from unittest.mock import patch

import torch
import torch.nn
import torch.utils.data
from pytest import approx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsifier
from tests.cross_fw.shared.paths import TEST_ROOT

FACTOR_NAME_IN_MOVEMENT_STAT = "movement_sparsity/importance_regularization_factor"
THRESHOLD_NAME_IN_MOVEMENT_STAT = "movement_sparsity/importance_threshold"
LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT = "movement_sparsity/linear_layer_sparsity"
MODEL_SPARSITY_NAME_IN_MOVEMENT_STAT = "movement_sparsity/model_sparsity"

TRAINING_SCRIPTS_PATH = TEST_ROOT.joinpath("torch", "sparsity", "movement", "training_scripts")
MRPC_CONFIG_FILE_NAME = "bert_tiny_uncased_mrpc_movement.json"


def mock_linear_nncf_node(
    in_features: int = 1, out_features: int = 1, bias: bool = True, node_name: str = "linear"
) -> NNCFNode:
    graph = NNCFGraph()
    linear = graph.add_nncf_node(
        node_name,
        "linear",
        Mock(),
        LinearLayerAttributes(
            weight_requires_grad=True, in_features=in_features, out_features=out_features, with_bias=bias
        ),
    )
    return linear


def initialize_sparsifier_parameters_by_linspace(
    operand: MovementSparsifier, linspace_start: float = -1, linspace_end: float = 1, seed: int = 42
):
    device = operand.weight_importance.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    with torch.no_grad():
        weight_rand_idx = torch.randperm(operand.weight_importance.numel(), generator=g, device=device)
        weight_init_tensor = torch.linspace(
            linspace_start, linspace_end, steps=operand.weight_importance.numel(), device=device
        )[weight_rand_idx].reshape_as(operand.weight_importance)
        operand.weight_importance.copy_(weight_init_tensor)
        if operand.prune_bias:
            bias_rand_idx = torch.randperm(operand.bias_importance.numel(), generator=g, device=device)
            bias_init_tensor = torch.linspace(
                linspace_start, linspace_end, steps=operand.bias_importance.numel(), device=device
            )[bias_rand_idx].reshape_as(operand.bias_importance)
            operand.bias_importance.copy_(bias_init_tensor)


def force_update_sparsifier_binary_masks_by_threshold(operand: MovementSparsifier, threshold: Optional[float] = None):
    if threshold is not None:
        operand.importance_threshold = threshold
    with patch.object(operand, "training", True), patch.object(operand, "frozen", False):
        operand._calc_training_binary_mask()
        if operand.prune_bias:
            operand._calc_training_binary_mask(is_bias=True)


def is_roughly_non_decreasing(x_list, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    x_list = list(x_list)
    assert rtol >= 0 and atol >= 0
    return all(a <= b + atol + rtol * abs(b) for a, b in zip(x_list[:-1], x_list[1:]))


def is_roughly_of_same_value(x_list, atol: float = 1e-6) -> bool:
    x_list = list(x_list)
    assert atol >= 0
    return all(x == approx(x_list[0], abs=atol) for x in x_list[1:])
