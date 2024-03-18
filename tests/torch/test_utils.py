# Copyright (c) 2024 Intel Corporation
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
from torch import nn

from nncf.torch.initialization import DataLoaderBNAdaptationRunner
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.utils import _ModuleState
from nncf.torch.utils import get_model_device
from nncf.torch.utils import get_model_dtype
from nncf.torch.utils import is_multidevice
from nncf.torch.utils import save_module_state
from nncf.torch.utils import training_mode_switcher
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import EmptyModel
from tests.torch.helpers import MockModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.quantization.test_overflow_issue_export import DepthWiseConvTestModel
from tests.torch.quantization.test_overflow_issue_export import EightConvTestModel


def compare_saved_model_state_and_current_model_state(model: nn.Module, model_state: _ModuleState):
    for name, module in model.named_modules():
        assert model_state.training_state[name] == module.training

    for name, param in model.named_parameters():
        assert param.requires_grad == model_state.requires_grad_state[name]


def randomly_change_model_state(module: nn.Module, compression_params_only: bool = False):
    import random

    for ch in module.modules():
        if random.uniform(0, 1) > 0.5:
            ch.training = False
        else:
            ch.training = True

    for p in module.parameters():
        if compression_params_only and not (isinstance(p, CompressionParameter) and torch.is_floating_point(p)):
            break
        if random.uniform(0, 1) > 0.5:
            p.requires_grad = False
        else:
            p.requires_grad = True


@pytest.mark.parametrize(
    "model", [BasicConvTestModel(), TwoConvTestModel(), MockModel(), DepthWiseConvTestModel(), EightConvTestModel()]
)
def test_training_mode_switcher(_seed, model: nn.Module):
    randomly_change_model_state(model)
    saved_state = save_module_state(model)
    with training_mode_switcher(model, True):
        pass

    compare_saved_model_state_and_current_model_state(model, saved_state)


@pytest.mark.parametrize(
    "model", [BasicConvTestModel(), TwoConvTestModel(), MockModel(), DepthWiseConvTestModel(), EightConvTestModel()]
)
def test_bn_training_state_switcher(_seed, model: nn.Module):
    def check_were_only_bn_training_state_changed(model: nn.Module, saved_state: _ModuleState):
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert module.training
            else:
                assert module.training == saved_state.training_state[name]

    runner = DataLoaderBNAdaptationRunner(model, "cuda")

    randomly_change_model_state(model)
    saved_state = save_module_state(model)

    with runner._bn_training_state_switcher():
        check_were_only_bn_training_state_changed(model, saved_state)

    compare_saved_model_state_and_current_model_state(model, saved_state)


def test_model_device():
    if not torch.cuda.is_available():
        return

    model = TwoConvTestModel()
    cuda = torch.device("cuda")

    assert not is_multidevice(model)
    assert get_model_device(model).type == "cpu"

    model.features[0][0].to(cuda)

    assert is_multidevice(model)
    assert get_model_device(model).type == "cuda"

    model.to(cuda)

    assert not is_multidevice(model)
    assert get_model_device(model).type == "cuda"


def test_empty_model_device():
    model = EmptyModel()

    assert not is_multidevice(model)
    assert get_model_device(model).type == "cpu"


def test_model_dtype():
    model = BasicConvTestModel()
    model.to(torch.float16)
    assert get_model_dtype(model) == torch.float16
    model.to(torch.float32)
    assert get_model_dtype(model) == torch.float32
    model.to(torch.float64)
    assert get_model_dtype(model) == torch.float64


def test_empty_model_dtype():
    model = EmptyModel()
    assert get_model_dtype(model) == torch.float32
