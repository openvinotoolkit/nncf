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
import logging

import pytest
import torch
from torch import nn
from torchvision import models

import nncf
from nncf.common.logging.logger import nncf_logger
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.magnitude.structured_modules import StructuredPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage


class EmbeddingLinearModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.tensor([[0, 1, 2]])

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.lin = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor):
        return self.lin(self.emb(x))


@pytest.fixture()
def nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


def check_all_two_four_groups(model: nn.Module) -> None:
    """
    Checks that every complete group of four consecutive weights within one output channel
    contains exactly two retained values, for each structured pruning mask in the model.
    """
    structured_hooks = [
        hook for _, hook in get_hook_storage(model).named_hooks() if isinstance(hook, StructuredPruningMask)
    ]
    assert structured_hooks
    for hook in structured_hooks:
        mask = hook.binary_mask
        grouped = mask.reshape(mask.shape[0], -1, 4)
        assert torch.all(grouped.sum(dim=-1) == 2)


def test_prune_2_4_resnet18_pattern():
    # resnet18 covers convolutions, depthwise-convolution shortcuts and Linear layers;
    # conv1 has 3 * 7 * 7 = 147 weights per output channel, which is not divisible by the
    # group size 4 of the 2:4 pattern, so it is excluded from pruning.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 200, bias=True)
    example_inputs = torch.rand(1, 3, 64, 64)
    pruned_model = nncf.prune(
        model,
        mode=PruneMode.STRUCTURED_MAGNITUDE_2_4,
        ignored_scope=nncf.IgnoredScope(names=["conv1/conv2d/0"]),
        examples_inputs=example_inputs,
    )

    hooks = dict(get_hook_storage(pruned_model).named_hooks())
    assert len(hooks) == 20
    assert all(isinstance(hook, StructuredPruningMask) for hook in hooks.values())
    check_all_two_four_groups(pruned_model)

    output = pruned_model(example_inputs)
    assert output.shape == (1, 200)


def test_prune_2_4_unsupported_op_skipped_with_warning(nncf_caplog):
    model = EmbeddingLinearModel()
    example_inputs = EmbeddingLinearModel.get_example_inputs()
    with nncf_caplog.at_level(logging.WARNING):
        pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)

    assert "not supported" in nncf_caplog.text
    assert "emb.weight" in nncf_caplog.text
    assert "PTEmbeddingMetatype" in nncf_caplog.text

    hooks = dict(get_hook_storage(pruned_model).named_hooks())
    assert list(hooks) == ["post_hooks.lin:weight__0.0"]
    check_all_two_four_groups(pruned_model)
    pruned_model(example_inputs)
