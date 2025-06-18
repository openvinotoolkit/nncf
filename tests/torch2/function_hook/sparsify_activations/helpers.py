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


from collections import defaultdict

import openvino as ov
import torch
import torch.nn as nn
import transformers.models

from nncf import IgnoredScope
from nncf.experimental.torch.sparsify_activations import TargetScope


class ThreeLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 2)
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(2, 4, bias=False)
        self.linear3 = nn.Linear(3, 5)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        y0 = self.linear3(self.linear1(x))
        y1 = self.linear2(x)
        return y0, y1


def dummy_llama_model():
    config = transformers.models.llama.configuration_llama.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=14,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        use_cache=False,
        return_dict=False,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return model


def count_sparsifier_patterns_in_ov(model: ov.Model) -> int:
    """
    Counts the number of activation sparsification pattern "Abs -> LessEqual -> Select"
    in the OpenVINO model.
    """
    pattern = ("Abs", "LessEqual", "Select")
    result = 0
    connections = defaultdict(list)
    for node in model.get_ops():
        for output in node.outputs():
            for input_ in output.get_target_inputs():
                connections[node].append(input_.get_node())

    def dfs(node, location=0):
        nonlocal result
        if location < len(pattern) and node.get_type_name() == pattern[location]:
            if location == len(pattern) - 1:
                result += 1
            else:
                for next_node in connections[node]:
                    dfs(next_node, location + 1)

    for node in model.get_ops():
        dfs(node)
    return result


def convert_ignored_scope_to_target_scope(ignored_scope: IgnoredScope) -> TargetScope:
    return TargetScope(
        ignored_scope.names,
        ignored_scope.patterns,
        ignored_scope.types,
        ignored_scope.subgraphs,
        ignored_scope.validate,
    )
