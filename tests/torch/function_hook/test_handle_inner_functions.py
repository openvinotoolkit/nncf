# Copyright (c) 2026 Intel Corporation
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
import torch.nn.functional as F
from torch import nn

from nncf.torch.function_hook.graph.build_graph_mode import build_graph
from nncf.torch.function_hook.graph.graph_visualization import to_pydot
from nncf.torch.function_hook.wrapper import wrap_model
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch" / "data" / "function_hook" / "handle_inner_functions"


class ReluModel(nn.Module):
    @staticmethod
    def example_input() -> tuple[torch.Tensor, ...]:
        return (torch.rand(1, 1, 1),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x)
        x = torch.relu_(x)
        x = F.relu(x, inplace=False)
        x = F.relu(x, inplace=True)
        return x


class BatchNormModel(nn.Module):
    @staticmethod
    def example_input() -> tuple[torch.Tensor, ...]:
        return (torch.rand(1, 1, 1),)

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        return x


class MultiHeadAttention(nn.Module):
    batch_size = 2
    seq_length = 5
    embed_dim = 16
    num_heads = 4

    @classmethod
    def example_input(cls) -> tuple[torch.Tensor, ...]:
        query = torch.rand((cls.seq_length, cls.batch_size, cls.embed_dim))
        key = torch.rand((cls.seq_length, cls.batch_size, cls.embed_dim))
        value = torch.rand((cls.seq_length, cls.batch_size, cls.embed_dim))
        return (query, key, value)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        in_proj_weight = torch.rand((3 * self.embed_dim, self.embed_dim))
        in_proj_bias = torch.rand(3 * self.embed_dim)
        bias_k = None
        bias_v = None
        add_zero_attn = False
        dropout_p = 0.1
        out_proj_weight = torch.rand((self.embed_dim, self.embed_dim))
        out_proj_bias = torch.rand(self.embed_dim)

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=bias_k,
            bias_v=bias_v,
            add_zero_attn=add_zero_attn,
            dropout_p=dropout_p,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            use_separate_proj_weight=False,
            training=True,
            need_weights=True,
        )


@pytest.mark.parametrize("model_cls", (ReluModel, BatchNormModel, MultiHeadAttention))
def test_inner_functions(model_cls: nn.Module, regen_ref_data: bool):
    model = wrap_model(model_cls()).eval()
    graph = build_graph(model, *model_cls.example_input())

    dot_graph = to_pydot(graph)
    ref_file = REF_DIR / f"inner_functions_{model_cls.__name__}.dot"
    compare_with_reference_file(str(dot_graph), ref_file, regen_ref_data)
