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

from typing import List, Union

import pytest
import torch
from pytest import FixtureRequest
from torch import nn

from nncf.experimental.torch2.function_hook.graph.build_graph_mode import GraphBuilderMode
from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import EdgeMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import NodeType
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorInfo
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorSource
from nncf.experimental.torch2.function_hook.hook_executor_mode import OpMeta
from nncf.experimental.torch2.function_hook.hook_storage import HookStorage
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.experimental.torch2.function_hook.wrapper import wrap_model
from tests.torch2.function_hook import helpers


def test_register_new_node_id():
    ctx = GraphBuilderMode(nn.Identity(), HookStorage())
    assert ctx.next_node_id == 0

    id = ctx.register_new_node_id()
    assert id == 0
    assert ctx.next_node_id == 1

    id = ctx.register_new_node_id()
    assert id == 1
    assert ctx.next_node_id == 2


def test_execute_hooks_for_model_input():
    ctx = GraphBuilderMode(nn.Identity(), HookStorage())

    tensor = torch.ones(1)
    ctx.execute_hooks_for_model_input("x", tensor)

    assert ctx.graph.nodes[0]["type"] == NodeType.input
    assert ctx.graph.nodes[0]["meta"] == InOutMeta(tensor.dtype, tuple(tensor.shape), "x")


def test_execute_hooks_for_model_output():
    ctx = GraphBuilderMode(nn.Identity(), HookStorage())

    tensor = torch.ones(1)
    node_id = ctx.register_new_node_id()
    ctx.tensor_info[tensor] = TensorInfo(
        NodeType.const,
        tuple(tensor.shape),
        tensor.dtype,
        0,
        node_id,
        None,
    )
    ctx.execute_hooks_for_model_output("output", tensor)

    assert ctx.graph.nodes[1]["type"] == NodeType.output
    assert ctx.graph.nodes[1]["meta"] == InOutMeta(tensor.dtype, tuple(tensor.shape), "output")

    assert ctx.graph.edges[0, 1, 0]["meta"] == EdgeMeta(dtype=torch.float32, shape=(1,), input_port=0, output_port=0)


def test_execute_pre_hooks():
    model = helpers.get_wrapped_simple_model_with_hook()
    ctx = GraphBuilderMode(model, get_hook_storage(model))

    ctx.execute_pre_hooks((torch.ones(1), model.conv.weight), {"b": model.conv.bias}, OpMeta("/relu/0", torch.relu))

    nodes = list(ctx.graph.nodes(data=True))
    assert nodes[0][0] == 0
    assert nodes[0][1] == {
        "type": NodeType.const,
        "meta": ConstMeta(dtype=torch.float32, shape=(1, 1, 1, 1), name_in_model="conv.weight"),
    }

    assert nodes[1][0] == 1
    assert nodes[1][1] == {
        "type": NodeType.const,
        "meta": ConstMeta(dtype=torch.float32, shape=(1,), name_in_model="conv.bias"),
    }

    assert nodes[2][0] == 2
    assert nodes[2][1] == {
        "type": NodeType.fn_call,
        "meta": FunctionMeta(
            op_name="/relu/0",
            fn_name="relu",
            args=(
                TensorMeta(dtype=torch.float32, shape=(1,), requires_grad=False),
                TensorMeta(dtype=torch.float32, shape=(1, 1, 1, 1), requires_grad=True),
            ),
            kwargs={"b": TensorMeta(dtype=torch.float32, shape=(1,), requires_grad=True)},
        ),
    }
    assert len(nodes) == 3

    edges = list(ctx.graph.edges(data=True))
    assert edges[0][0] == 0
    assert edges[0][1] == 2
    assert edges[0][2] == {"meta": EdgeMeta(dtype=torch.float32, shape=(1, 1, 1, 1), input_port=1, output_port=0)}

    assert edges[1][0] == 1
    assert edges[1][1] == 2
    assert edges[1][2] == {"meta": EdgeMeta(dtype=torch.float32, shape=(1,), input_port=2, output_port=0)}
    assert len(edges) == 2


@pytest.fixture(params=["tensor", "list", "torch_return_type"])
def example_outputs(request: FixtureRequest) -> Union[torch.Tensor, List[torch.Tensor], torch.return_types.max]:
    return {
        "tensor": torch.tensor(1.0),
        "list": [torch.tensor(1), torch.tensor([2])],
        "torch_return_type": torch.return_types.max((torch.tensor(1.0), torch.tensor([2]))),
    }.get(request.param)


def test_execute_post_hooks(example_outputs: Union[torch.Tensor, List[torch.Tensor], torch.return_types.max]):
    ctx = GraphBuilderMode(nn.Identity(), HookStorage())
    op_meta = OpMeta("/relu/0", torch.relu, {"node_id": 0})
    ctx.execute_post_hooks(example_outputs, op_meta)

    if isinstance(example_outputs, torch.Tensor):
        assert ctx.tensor_info[example_outputs] == TensorInfo(TensorSource.function, (), torch.float32, 0, 0, None)

    if isinstance(example_outputs, list):
        assert ctx.tensor_info[example_outputs[0]] == TensorInfo(TensorSource.function, (), torch.int64, 0, 0, None)
        assert ctx.tensor_info[example_outputs[1]] == TensorInfo(TensorSource.function, (1,), torch.int64, 1, 0, None)

    if isinstance(example_outputs, torch.return_types.max):
        assert ctx.tensor_info[example_outputs.values] == TensorInfo(
            TensorSource.function, (), torch.float32, 0, 0, None
        )
        assert ctx.tensor_info[example_outputs.indices] == TensorInfo(
            TensorSource.function, (1,), torch.int64, 1, 0, None
        )


def test_build_graph():
    model = helpers.get_wrapped_simple_model_with_hook()
    graph = build_graph(model, model.get_example_inputs())

    nodes = list(graph.nodes(data=True))
    assert len(nodes) == 8
    assert len(list(filter(lambda x: x[1]["type"] == NodeType.input, nodes))) == 1
    assert len(list(filter(lambda x: x[1]["type"] == NodeType.output, nodes))) == 1
    assert len(list(filter(lambda x: x[1]["type"] == NodeType.const, nodes))) == 3
    assert len(list(filter(lambda x: x[1]["type"] == NodeType.fn_call, nodes))) == 3

    edges = list(graph.edges(data=True))
    assert len(edges) == 7
    assert len(list(filter(lambda x: isinstance(x[2]["meta"], EdgeMeta), edges))) == 7


class ModelTensorAttribute(nn.Module):
    def __init__(self, attribute: str) -> None:
        super().__init__()
        self.attr = attribute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attr == ".T":
            return x.T
        if self.attr == ".mT":
            return x.mT
        raise ValueError(f"Unexpected attribute: {self.attr}")


@pytest.mark.parametrize("attr", [".T", ".mT"])
def test_tensor_attributes(attr):
    model = ModelTensorAttribute(attribute=attr)
    model = wrap_model(model)
    graph = build_graph(model, torch.ones(2, 3, requires_grad=True))

    if attr == ".T":
        ref_meta = FunctionMeta(
            op_name="/__get__/0",
            fn_name="permute",
            args=(TensorMeta(dtype=torch.float32, shape=(2, 3), requires_grad=True),),
            kwargs={"dims": (1, 0)},
        )
    else:
        ref_meta = FunctionMeta(
            op_name="/__get__/0",
            fn_name="transpose",
            args=(TensorMeta(dtype=torch.float32, shape=(2, 3), requires_grad=True),),
            kwargs={"dim0": -2, "dim1": -1},
        )
    assert graph.nodes[1]["type"] == NodeType.fn_call
    assert graph.nodes[1]["meta"] == ref_meta


class ModelTensorAttribute2(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        _ = x.shape
        x = x.T
        x = x + 1
        _ = x.shape
        return x


class HookWithAttribute(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = x.ndim
        x = x - 1
        return x


def test_remove_get_node_with_no_tensor_output():
    model = ModelTensorAttribute2()
    model = wrap_model(model)
    hook_storage = get_hook_storage(model)
    hook_storage.register_pre_function_hook("/add/1", 0, HookWithAttribute())

    # Set requires_grad=True to get grad_fn
    graph = build_graph(model, torch.ones(2, 3, requires_grad=True))

    nodes = list(graph.nodes(data=True))
    get_op_nodes = list(
        filter(lambda x: isinstance(x[1]["meta"], FunctionMeta) and "__get__" in x[1]["meta"].op_name, nodes)
    )
    assert len(get_op_nodes) == 1
    assert get_op_nodes[0][1]["meta"].op_name == "/__get__/1"


def test_multi_edges():
    model = helpers.ModelMultiEdge()
    model = wrap_model(model)
    graph = build_graph(model, torch.ones(1, 1))

    edges = list(graph.edges(data=True))
    ref_edges = [
        (0, 1, {"meta": EdgeMeta(dtype=torch.float32, shape=(1, 1), input_port=0, output_port=0)}),
        (0, 1, {"meta": EdgeMeta(dtype=torch.float32, shape=(1, 1), input_port=1, output_port=0)}),
        (1, 2, {"meta": EdgeMeta(dtype=torch.float32, shape=(1, 1), input_port=0, output_port=0)}),
    ]
    assert edges == ref_edges
