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

from pathlib import Path

import pytest
import torch
from torch import nn

import nncf
import nncf.torch
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.magnitude.modules import UnstructuredPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from tests.torch2.function_hook.pruning.helpers import ConvModel
from tests.torch2.function_hook.pruning.helpers import MatMulLeft
from tests.torch2.function_hook.pruning.helpers import MatMulRight
from tests.torch2.function_hook.pruning.helpers import MultiDeviceModel
from tests.torch2.function_hook.pruning.helpers import SharedParamModel
from tests.torch2.function_hook.pruning.helpers import TwoConvModel


@pytest.mark.parametrize(
    "model_cls, ref",
    (
        (ConvModel, "post_hooks.conv:weight__0.0"),
        (MatMulLeft, "post_hooks.w__0.0"),
        (MatMulRight, "post_hooks.w__0.0"),
        (SharedParamModel, "post_hooks.module1:0:weight__0.0"),
    ),
)
def test_prune_model(model_cls: nn.Module, ref: str):
    model = model_cls()
    example_inputs = model_cls.get_example_inputs()
    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs
    )
    hook_storage = get_hook_storage(pruned_model)

    for name, sparsity_module in hook_storage.named_hooks():
        assert name == ref
        assert isinstance(sparsity_module, UnstructuredPruningMask)
        assert sparsity_module.binary_mask.dtype == torch.bool


@pytest.mark.parametrize(
    "mode, ref",
    (
        (
            PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL,
            {
                "post_hooks.conv1:weight__0.0": [False, False, True, True],
                "post_hooks.conv2:weight__0.0": [False, False, True, True],
            },
        ),
        (
            PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
            {
                "post_hooks.conv1:weight__0.0": [False, True, True, True],
                "post_hooks.conv2:weight__0.0": [False, False, False, True],
            },
        ),
    ),
)
def test_prune_mode(mode: PruneMode, ref):
    model = TwoConvModel()
    example_inputs = TwoConvModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=mode, ratio=0.5, examples_inputs=example_inputs)
    hook_storage = get_hook_storage(pruned_model)
    for name, sparsity_module in hook_storage.named_hooks():
        assert isinstance(sparsity_module, UnstructuredPruningMask)
        c = sparsity_module.binary_mask.view(-1).tolist()
        assert c == ref[name]


def test_infer(use_cuda: bool):
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()

    if use_cuda:
        model = model.cuda()
        example_inputs = example_inputs.cuda()

    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs
    )
    pruned_model(example_inputs)


@pytest.mark.cuda
def test_multi_device_infer():
    model = MultiDeviceModel()
    example_inputs = MultiDeviceModel.get_example_inputs()

    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs
    )
    pruned_model(example_inputs)

    hook_storage = get_hook_storage(pruned_model)

    ref_devices = {
        "post_hooks.conv1:weight__0.0": "cpu",
        "post_hooks.conv2:weight__0.0": "cuda",
    }
    act_devices = {}
    for name, sparsity_module in hook_storage.named_hooks():
        assert isinstance(sparsity_module, UnstructuredPruningMask)
        act_devices[name] = sparsity_module.binary_mask.device.type
    assert ref_devices == act_devices


def test_save_load(tmpdir: Path):
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()

    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs
    )
    checkpoint = {
        "state_dict": pruned_model.state_dict(),
        "nncf_config": nncf.torch.get_config(pruned_model),
    }
    path_to_checkpoint = tmpdir / "checkpoint.pth"
    torch.save(checkpoint, path_to_checkpoint)

    resuming_checkpoint = torch.load(path_to_checkpoint)
    nncf_config = resuming_checkpoint["nncf_config"]
    state_dict = resuming_checkpoint["state_dict"]

    loaded_model = ConvModel()
    loaded_pruned_model = nncf.torch.load_from_config(loaded_model, nncf_config, example_inputs)
    loaded_pruned_model.load_state_dict(state_dict)

    hook_storage = get_hook_storage(loaded_pruned_model)

    d = {k: v for k, v in hook_storage.named_hooks()}
    assert len(d) == 1
    assert isinstance(d["post_hooks.conv:weight__0.0"], UnstructuredPruningMask)
