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
from packaging import version

from nncf.torch import wrap_model
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.trace_tensor import TracedParameter
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.wrappers import wrap_parameters
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from tests.torch.helpers import BasicConvTestModel


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.11"),
    reason="__getitem__ works unexpectedly for TracedTensor until fix in torch 1.11.\n"
    "Fix in pytorch: https://github.com/pytorch/pytorch/pull/67202\n"
    "Related ticket: 82065",
)
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

    assert not ctx._threading.thread_local.operator_counters


class ModuleForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, 1)
        self.cached_tensor = None
        self.weight = torch.nn.Parameter(torch.zeros([1, 2]))

    def forward(self, x):
        if self.cached_tensor is not None:
            # self.cached_tensor is produced from a later
            # stage in control flow graph, but on previous forward
            self.cached_tensor.unsqueeze(0)
        x = self.conv2d(x)
        self.cached_tensor = x
        return self.conv2d(x)


def test_traced_tensors_are_stripped_on_context_exit():
    module = ModuleForTest()
    module.train()
    tensor = torch.ones([1, 1, 1, 1])
    with TracingContext():
        wrap_parameters(module)
        result = module(tensor)
        assert isinstance(module.cached_tensor, TracedTensor)
        assert isinstance(module.weight, TracedParameter)
        assert isinstance(module.conv2d.weight, TracedParameter)
        assert isinstance(result, TracedTensor)
    assert isinstance(module.cached_tensor, torch.Tensor)
    assert isinstance(result, torch.Tensor)
    assert isinstance(module.weight, torch.nn.Parameter)
    assert isinstance(module.conv2d.weight, torch.nn.Parameter)


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


@pytest.mark.parametrize(
    "contexts",
    [3 * [TracingContext()], [TracingContext(), TracingContext(), TracingContext()]],
    ids=["same", "different"],
)
def test_nested_contexts(contexts):
    module = ModuleForTest()
    module.train()
    tensor = torch.ones([1, 1, 1, 1])
    nesting_count = [1]
    with contexts[0]:
        with contexts[1]:
            count = contexts[:2].count(contexts[1])
            nesting_count.append(count)
            with contexts[2]:
                count = contexts.count(contexts[2])
                nesting_count.append(count)
                module(tensor)
                assert len(contexts[2]._threading.thread_local.nested_contexts_stack) == nesting_count[2]
                assert len(contexts[2]._threading.thread_local.traced_tensor_weakrefs) > 0
            assert len(contexts[1]._threading.thread_local.nested_contexts_stack) == nesting_count[1]
            assert len(contexts[1]._threading.thread_local.traced_tensor_weakrefs) > 0
        assert len(contexts[0]._threading.thread_local.nested_contexts_stack) == nesting_count[0]
        assert contexts[0]._threading.thread_local.nested_contexts_stack[0] is None
        assert len(contexts[0]._threading.thread_local.traced_tensor_weakrefs) > 0
    assert len(contexts[0]._threading.thread_local.traced_tensor_weakrefs) == 0


@pytest.mark.parametrize("compression_model_type", ExtraCompressionModuleType)
def test_not_trace_parameters_in_nncf_modules(compression_model_type):
    model = wrap_model(BasicConvTestModel(), torch.ones(BasicConvTestModel.INPUT_SIZE), trace_parameters=True)
    model.nncf.register_compression_module_type(compression_model_type)
    model.nncf.add_compression_module("test", torch.nn.Conv2d(1, 1, 1), compression_model_type)

    with TracingContext() as ctx:
        wrap_parameters(model)
        assert len(ctx._threading.thread_local.traced_tensor_weakrefs) == 2
