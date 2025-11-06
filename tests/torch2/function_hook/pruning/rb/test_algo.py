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
from nncf.torch.function_hook.pruning.rb.modules import RBPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from tests.torch2.function_hook.pruning.helpers import ConvModel
from tests.torch2.function_hook.pruning.helpers import MatMulLeft
from tests.torch2.function_hook.pruning.helpers import MatMulRight
from tests.torch2.function_hook.pruning.helpers import MultiDeviceModel
from tests.torch2.function_hook.pruning.helpers import SharedParamModel


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
    pruned_model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, examples_inputs=example_inputs)
    hook_storage = get_hook_storage(pruned_model)

    for name, sparsity_module in hook_storage.named_hooks():
        assert name == ref
        assert isinstance(sparsity_module, RBPruningMask)
        assert sparsity_module.mask.dtype == torch.float32


def test_infer(use_cuda: bool):
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()

    if use_cuda:
        model = model.cuda()
        example_inputs = example_inputs.cuda()

    pruned_model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, examples_inputs=example_inputs)
    pruned_model(example_inputs)


@pytest.mark.cuda
def test_multi_device_infer():
    model = MultiDeviceModel()
    example_inputs = MultiDeviceModel.get_example_inputs()

    pruned_model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, examples_inputs=example_inputs)
    pruned_model(example_inputs)

    hook_storage = get_hook_storage(pruned_model)

    ref_devices = {
        "post_hooks.conv1:weight__0.0": "cpu",
        "post_hooks.conv2:weight__0.0": "cuda",
    }
    act_devices = {}
    for name, sparsity_module in hook_storage.named_hooks():
        assert isinstance(sparsity_module, RBPruningMask)
        act_devices[name] = sparsity_module.mask.device.type
    assert ref_devices == act_devices


def test_save_load(tmpdir: Path):
    model = ConvModel().eval()
    example_inputs = ConvModel.get_example_inputs()

    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, ratio=0.5, examples_inputs=example_inputs
    )
    pruned_model.eval()
    checkpoint = {
        "state_dict": pruned_model.state_dict(),
        "nncf_config": nncf.torch.get_config(pruned_model),
    }
    path_to_checkpoint = tmpdir / "checkpoint.pth"
    torch.save(checkpoint, path_to_checkpoint)
    orig_output = pruned_model(example_inputs)

    resuming_checkpoint = torch.load(path_to_checkpoint)
    nncf_config = resuming_checkpoint["nncf_config"]
    state_dict = resuming_checkpoint["state_dict"]

    loaded_model = ConvModel()
    loaded_pruned_model = nncf.torch.load_from_config(loaded_model, nncf_config, example_inputs)
    loaded_pruned_model.load_state_dict(state_dict)
    loaded_pruned_model.eval()
    loaded_output = loaded_pruned_model(example_inputs)

    hook_storage = get_hook_storage(loaded_pruned_model)

    d = {k: v for k, v in hook_storage.named_hooks()}
    assert len(d) == 1
    assert isinstance(d["post_hooks.conv:weight__0.0"], RBPruningMask)

    assert torch.allclose(orig_output, loaded_output)


def test_statistic():
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()

    pruned_model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, ratio=0.5, examples_inputs=example_inputs
    )

    # Set mask
    with torch.no_grad():
        hook_storage = get_hook_storage(pruned_model)
        pruning_module = hook_storage.post_hooks["conv:weight__0"]["0"]
        pruning_module.mask[0] *= -1

    stat = nncf.pruning_statistic(pruned_model)

    assert pytest.approx(stat.pruned_tensors[0].pruned_ratio, abs=1e-1) == 0.3
    assert stat.pruned_tensors[0].tensor_name == "conv.weight"
    assert stat.pruned_tensors[0].shape == (3, 3, 3, 3)
    assert pytest.approx(stat.pruning_ratio, abs=1e-2) == 0.33
    assert pytest.approx(stat.global_pruning_ratio, abs=1e-2) == 0.32

    txt = str(stat)
    assert "conv.weight" in txt
    assert "All parameters" in txt
