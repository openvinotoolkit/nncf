"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import pytest
import torch
from pkg_resources import parse_version

from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor


@pytest.mark.skipif(parse_version(torch.__version__) < parse_version("1.11"),
                    reason="__getitem__ works unexpectedly for TracedTensor until fix in torch 1.11.\n"
                           "Fix in pytorch: https://github.com/pytorch/pytorch/pull/67202\n"
                           "Related ticket: 82065")
def test_torch_tensor_getitem_behavior(mocker):
    x = torch.ones((10, 4, 4, 4))
    indexes = torch.LongTensor([0, 1, 2])
    mock_tensor_meta = mocker.stub
    traced_x = TracedTensor.from_torch_tensor(torch.ones((10, 4, 4, 4)), mock_tensor_meta)
    traced_indexes = TracedTensor.from_torch_tensor(torch.LongTensor([0, 1, 2]), mock_tensor_meta)
    SHAPE_1 = [3, 4, 4, 4]
    SHAPE_2 = [3, 4, 4]
    assert list(x[indexes].shape) == SHAPE_1
    assert list(x[traced_indexes].shape) == SHAPE_1
    assert list(traced_x[indexes].shape) == SHAPE_1
    assert list(traced_x[traced_indexes].shape) == SHAPE_1

    assert list(x[indexes, indexes].shape) == SHAPE_2
    assert list(x[traced_indexes, traced_indexes].shape) == SHAPE_2
    assert list(traced_x[indexes, indexes].shape) == SHAPE_2
    assert list(traced_x[traced_indexes, traced_indexes].shape)
    assert list(x[indexes, traced_indexes].shape) == SHAPE_2
    assert list(traced_x[indexes, traced_indexes].shape) == SHAPE_2


class ModelSpecificException(Exception):
    pass


class ExceptionRaisingModule(torch.nn.Module):
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones([1]))

        def forward(self, *args, **kwargs):
            _ = torch.cat((torch.ones([1]), torch.ones([1])))
            raise ModelSpecificException

    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(self.Inner(), self.Inner())

    def forward(self, *args, **kwargs):
        return self.seq(*args, **kwargs)


def test_scope_and_call_counters_are_reset_on_exceptions():
    ctx = TracingContext()
    model = ExceptionRaisingModule()
    with pytest.raises(ModelSpecificException):
        with ctx:
            model(torch.ones([1]))
    assert not ctx.module_call_stack
    assert not ctx.relative_scopes_stack
    #pylint:disable=protected-access
    assert not ctx._threading.thread_local.operator_counters


class ModuleForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, 1)
        self.cached_tensor = None

    def forward(self, x):
        if self.cached_tensor is not None:
            # self.cached_tensor is produced from a later
            # stage in control flow graph, but on previous forward
            self.cached_tensor.unsqueeze(0)
        x = self.conv2d(x)
        self.cached_tensor = x
        return self.conv2d(x)


def test_traced_tensors_are_expired_on_context_exit():
    module = ModuleForTest()
    module.train()
    tensor = torch.ones([1, 1, 1, 1])
    with TracingContext():
        result = module(tensor)
    assert isinstance(module.cached_tensor, TracedTensor)
    assert module.cached_tensor.nncf_expired
    assert isinstance(result, TracedTensor)
    assert result.nncf_expired


def test_no_cross_forward_run_dependency():
    module = ModuleForTest()
    module.train()
    tensor = torch.ones([1, 1, 1, 1])
    with TracingContext() as ctx:
        ctx.enable_trace_dynamic_graph()
        _ = module(tensor)
        ctx.disable_trace_dynamic_graph()
    module.eval()
    with TracingContext() as ctx:
        ctx.enable_trace_dynamic_graph()
        _ = module(tensor)
        ctx.disable_trace_dynamic_graph()
