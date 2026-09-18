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

import nncf
from nncf.parameters import PruneMode
from nncf.scopes import IgnoredScope
from nncf.torch.function_hook.pruning.magnitude.structured_modules import StructuredPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from tests.torch.function_hook.pruning.helpers import TwoConvModel


class TwoLinearModel(torch.nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 8)

    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(8, 8, bias=False)
        self.lin2 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor):
        return self.lin2(torch.relu(self.lin1(x)))


class DivisibleConvModel(torch.nn.Module):
    # 8 * 3 * 3 = 72 weights per output channel, divisible by 4
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 8, 8, 8)

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class NonDivisibleConvModel(torch.nn.Module):
    # 3 * 3 * 3 = 27 weights per output channel, not divisible by 4
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 3, 8, 8)

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class NonDivisibleLinearModel(torch.nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 7)

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(7, 8, bias=False)

    def forward(self, x: torch.Tensor):
        return self.lin(x)


def get_binary_masks(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    hook_storage = get_hook_storage(model)
    return {name: hook.binary_mask for name, hook in hook_storage.named_hooks()}


def get_structured_hook(model: torch.nn.Module) -> StructuredPruningMask:
    hook_storage = get_hook_storage(model)
    hooks = [hook for _, hook in hook_storage.named_hooks()]
    assert len(hooks) == 1
    assert isinstance(hooks[0], StructuredPruningMask)
    return hooks[0]


def check_two_four_groups(mask: torch.Tensor) -> None:
    """
    Checks that every complete group of four consecutive weights within one output channel
    contains exactly two retained values, following the output-channel-by-reduction layout
    used by the structured pruning implementation.
    """
    num_out_channels = mask.shape[0]
    grouped = mask.reshape(num_out_channels, -1, 4)
    group_sums = grouped.sum(dim=-1)
    assert torch.all(group_sums == 2)


@pytest.mark.parametrize("pruned_op_name, other_op_name", (("lin1", "lin2"), ("lin2", "lin1")))
def test_prune_2_4_linear_two_layers(pruned_op_name: str, other_op_name: str):
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)

    hooks = dict(get_hook_storage(pruned_model).named_hooks())
    assert f"post_hooks.{pruned_op_name}:weight__0.0" in hooks
    assert f"post_hooks.{other_op_name}:weight__0.0" in hooks
    for hook in hooks.values():
        assert isinstance(hook, StructuredPruningMask)
        check_two_four_groups(hook.binary_mask)

    pruned_model(example_inputs)


def test_prune_2_4_conv():
    model = DivisibleConvModel()
    example_inputs = DivisibleConvModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)

    hook = get_structured_hook(pruned_model)
    # For convolution weights the groups follow the output-channel-by-spatial-position layout:
    # every complete group of four consecutive weights within one output channel contains two retained values
    check_two_four_groups(hook.binary_mask)
    pruned_model(example_inputs)


def test_prune_2_4_conv_in_channels_not_divisible():
    model = NonDivisibleConvModel()
    with pytest.raises(nncf.ValidationError, match="not divisible"):
        nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=model.get_example_inputs())


def test_prune_2_4_in_features_not_divisible():
    model = NonDivisibleLinearModel()
    with pytest.raises(nncf.ValidationError, match="not divisible"):
        nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=model.get_example_inputs())


def test_prune_2_4_ignored_scope():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    ignored_scope = IgnoredScope(names=["lin1/linear/0"])
    pruned_model = nncf.prune(
        model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, ignored_scope=ignored_scope, examples_inputs=example_inputs
    )
    hooks = dict(get_hook_storage(pruned_model).named_hooks())
    assert "post_hooks.lin2:weight__0.0" in hooks
    assert "post_hooks.lin1:weight__0.0" not in hooks


def test_prune_2_4_ratio_is_ignored():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, ratio=0.9, examples_inputs=example_inputs)
    for hook in get_hook_storage(pruned_model).named_hooks():
        assert isinstance(hook[1], StructuredPruningMask)
        # A ratio of 0.9 cannot be expressed with a fixed 2:4 pattern
        check_two_four_groups(hook[1].binary_mask)


def test_prune_2_4_two_conv_models_keeps_unstructured_behaviour():
    # Unstructured magnitude pruning must keep using UnstructuredPruningMask
    model = TwoConvModel()
    example_inputs = TwoConvModel.get_example_inputs()
    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs
    )
    for _, hook in get_hook_storage(pruned_model).named_hooks():
        assert not isinstance(hook, StructuredPruningMask)


def test_prune_2_4_statistic():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)
    stat = nncf.pruning_statistic(pruned_model)
    assert stat.pruned_tensors[0].shape == (8, 8)
    assert pytest.approx(stat.pruning_ratio, abs=1e-9) == 0.5


def test_prune_2_4_strip():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)
    output_with_hooks = pruned_model(example_inputs)

    stripped_model = nncf.strip(pruned_model, example_input=example_inputs, strip_format=nncf.StripFormat.IN_PLACE)
    assert not list(get_hook_storage(stripped_model).named_hooks())
    for name, param in stripped_model.named_parameters():
        if "weight" in name:
            grouped = param.detach().reshape(param.shape[0], -1, 4)
            assert torch.all((grouped != 0).sum(dim=-1) == 2)
    assert torch.allclose(output_with_hooks, stripped_model(example_inputs))


def test_prune_2_4_save_load():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)
    original_output = pruned_model(example_inputs)

    config = nncf.torch.get_config(pruned_model)
    restored_model = nncf.torch.load_from_config(TwoLinearModel(), config)
    restored_model.load_state_dict(pruned_model.state_dict())
    restored_output = restored_model(example_inputs)
    assert torch.allclose(original_output, restored_output)

    for _, hook in get_hook_storage(restored_model).named_hooks():
        assert isinstance(hook, StructuredPruningMask)


def test_magnitude_scheduler_rejects_structured_mode():
    model = TwoLinearModel()
    example_inputs = TwoLinearModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, examples_inputs=example_inputs)
    from nncf.torch.function_hook.pruning.magnitude.schedulers import MultiStepMagnitudePruningScheduler

    with pytest.raises(nncf.ValidationError, match="Ratio schedulers are not supported"):
        MultiStepMagnitudePruningScheduler(
            model=pruned_model, mode=PruneMode.STRUCTURED_MAGNITUDE_2_4, steps={0: 0.5, 1: 0.7}
        )
