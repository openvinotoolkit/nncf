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
from torch import nn

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.utils.os import is_windows
from nncf.torch.initialization import DataLoaderBNAdaptationRunner
from nncf.torch.utils import CompilationWrapper
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


def change_model_state(module: nn.Module):
    for i, ch in enumerate(module.modules()):
        ch.training = i % 2 == 0

    for i, p in enumerate(module.parameters()):
        p.requires_grad = i % 2 == 0


@pytest.mark.parametrize(
    "model", [BasicConvTestModel(), TwoConvTestModel(), MockModel(), DepthWiseConvTestModel(), EightConvTestModel()]
)
def test_training_mode_switcher(model: nn.Module):
    change_model_state(model)
    saved_state = save_module_state(model)
    with training_mode_switcher(model, True):
        pass

    compare_saved_model_state_and_current_model_state(model, saved_state)


@pytest.mark.parametrize(
    "model", [BasicConvTestModel(), TwoConvTestModel(), MockModel(), DepthWiseConvTestModel(), EightConvTestModel()]
)
def test_bn_training_state_switcher(model: nn.Module):
    def check_were_only_bn_training_state_changed(model: nn.Module, saved_state: _ModuleState):
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert module.training
            else:
                assert module.training == saved_state.training_state[name]

    runner = DataLoaderBNAdaptationRunner(model, "cuda")

    for p in model.parameters():
        p.requires_grad = False

    saved_state = save_module_state(model)

    with runner._bn_training_state_switcher():
        check_were_only_bn_training_state_changed(model, saved_state)

    compare_saved_model_state_and_current_model_state(model, saved_state)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available in current environment")
def test_model_device():
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


def compilable_fn(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a - b


def not_compilable_fn(x, y):
    if torch.compiler.is_compiling():
        msg = "Controlled exception!"
        nncf_logger.debug(msg)
        raise nncf.InternalError(msg)
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


@pytest.mark.parametrize(
    "fn, is_compilation_successful",
    [
        (compilable_fn, True),
        (not_compilable_fn, False),
    ],
)
def test_compilation_wrapper(fn, is_compilation_successful):
    successful_ref = False if is_windows() else is_compilation_successful
    torch.compiler.reset()
    wrapped_fn = CompilationWrapper(fn)
    wrapped_fn(torch.randn(10, 10), torch.randn(10, 10))
    assert wrapped_fn.is_compilation_successful == successful_ref
    torch.compiler.reset()
