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

from dataclasses import dataclass
from functools import partial
from typing import Callable

import networkx as nx
import pytest
import torch
import torch.nn.functional as F
import torchvision.models as models

from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.function_hook.graph.build_graph_mode import build_graph
from nncf.torch.function_hook.graph.graph_utils import ConstMeta
from nncf.torch.function_hook.graph.graph_utils import FunctionMeta
from nncf.torch.function_hook.graph.graph_utils import InOutMeta
from nncf.torch.function_hook.graph.graph_utils import NodeType
from nncf.torch.function_hook.graph.graph_utils import TensorMeta
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import convert_to_nncf_graph
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import get_dtype
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import get_name_of_node
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import get_node_type
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.function_hook import helpers
from tests.torch.utils import compare_with_reference_file
from tests.torch.utils import to_comparable_nx_graph

REF_DIR = TEST_ROOT / "torch" / "data" / "function_hook" / "nncf_graph"


@pytest.mark.parametrize(
    "node_type, meta, ref",
    [
        [NodeType.input, InOutMeta(torch.float32, (1), "input"), "nncf_model_input"],
        [NodeType.output, InOutMeta(torch.float32, (1), "output"), "nncf_model_output"],
        [NodeType.output, FunctionMeta("op", torch.relu, [], {}), "relu"],
        [NodeType.output, ConstMeta(torch.float32, (1), "model.bias"), "nncf_model_const"],
    ],
)
def test_get_node_type(node_type: NodeType, meta: ConstMeta | FunctionMeta | InOutMeta, ref: str):
    assert get_node_type(node_type, meta) == ref


@pytest.mark.parametrize(
    "meta, ref",
    [
        (InOutMeta(torch.float32, (1), "input"), "input"),
        (InOutMeta(torch.float32, (1), "output"), "output"),
        (FunctionMeta("op_name", "fn_name", [], {}), "op_name"),
        (ConstMeta(torch.float32, (1), "model.bias"), "model.bias"),
    ],
)
def test_get_name_of_node(meta: InOutMeta | FunctionMeta | ConstMeta, ref: str):
    assert get_name_of_node(meta) == ref


@pytest.mark.parametrize(
    "dtype, ref",
    [
        [torch.float, Dtype.FLOAT],
        [torch.float32, Dtype.FLOAT],
        [torch.float64, Dtype.FLOAT],
        [torch.bfloat16, Dtype.FLOAT],
        [torch.int, Dtype.INTEGER],
        [torch.int8, Dtype.INTEGER],
        [torch.int16, Dtype.INTEGER],
        [torch.int32, Dtype.INTEGER],
        [torch.int64, Dtype.INTEGER],
    ],
)
def test_get_dtype(dtype: torch.dtype, ref: Dtype):
    assert get_dtype(dtype) == ref


def test_convert_to_nncf_graph(regen_ref_data: bool):
    model = helpers.get_wrapped_simple_model_with_hook()
    nx_graph = build_graph(model, model.get_example_inputs())

    nncf_graph = convert_to_nncf_graph(nx_graph)
    graph = to_comparable_nx_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)

    ref_file = REF_DIR / "convert_to_nncf_graph.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


def test_convert_to_nncf_graph_multi_edges(regen_ref_data: bool):
    model = helpers.ModelMultiEdge()
    model = wrap_model(model)
    nx_graph = build_graph(model, torch.ones(1, 1))
    nncf_graph = convert_to_nncf_graph(nx_graph)
    graph = to_comparable_nx_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / "convert_to_nncf_graph_multi_edges.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


@dataclass
class ModelDesc:
    model_name: str
    model_builder: callable
    inputs_info: list[list[int]] | tuple[list[int], ...]

    def __str__(self):
        return self.model_name


TEST_MODELS_DESC = [
    ModelDesc("convnext_small", partial(models.convnext_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("densenet121", partial(models.densenet121, weights=None), [1, 3, 64, 64]),
    ModelDesc("efficientnet_b0", partial(models.efficientnet_b0, weights=None), [1, 3, 64, 64]),
    ModelDesc("inception_v3", partial(models.inception_v3, init_weights=False, weights=None), [1, 3, 300, 300]),
    ModelDesc("mobilenet_v2", partial(models.mobilenet_v2, weights=None), [1, 3, 64, 64]),
    ModelDesc("mobilenet_v3_small", partial(models.mobilenet_v3_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnet18", partial(models.resnet18, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnext50_32x4d", partial(models.resnext50_32x4d, weights=None), [1, 3, 64, 64]),
    ModelDesc("shufflenet_v2_x0_5", partial(models.shufflenet_v2_x0_5, weights=None), [1, 3, 224, 224]),
    ModelDesc("squeezenet1_0", partial(models.squeezenet1_0, weights=None), [1, 3, 64, 64]),
    ModelDesc("swin_v2_b", partial(models.swin_v2_b, weights=None), [1, 3, 64, 64]),
    ModelDesc("vgg16", partial(models.vgg16, weights=None), [1, 3, 32, 32]),
    ModelDesc("gru", helpers.ModelGRU, [1, 3, 3]),
]


@pytest.mark.parametrize("desc", TEST_MODELS_DESC, ids=str)
def test_model_graph(desc: ModelDesc, regen_ref_data: bool):
    model: torch.nn.Module = desc.model_builder()
    model = model.eval()
    nncf_graph = build_nncf_graph(model, torch.randn(desc.inputs_info))
    graph = to_comparable_nx_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / f"model_graph_{desc}.dot"
    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


def test_model_graph_with_shared_parameters(regen_ref_data):
    model = wrap_model(helpers.SharedParamModel())
    register_post_function_hook(model, "module1:0:weight", 0, helpers.CounterHook())
    nncf_graph = build_nncf_graph(model, model.get_example_inputs())
    graph = to_comparable_nx_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / "model_graph_with_shared_parameters.dot"
    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


def _missed_input_edge_for_conv() -> PTNNCFGraph:
    graph = PTNNCFGraph()
    graph.add_nncf_node(
        node_name="conv",
        node_type="conv",
        node_metatype=PTConv2dMetatype,
        layer_attributes=PT2OpLayerAttributes(
            func=F.conv2d,
            op_args=(
                TensorMeta(shape=(1,), dtype=torch.float),
                TensorMeta(shape=(1,), dtype=torch.float),
            ),
            op_kwargs={},
            constant_port_ids=set([1]),
        ),
    )
    return graph


def _missed_input_edge_for_concat() -> PTNNCFGraph:
    graph = PTNNCFGraph()
    graph.add_nncf_node(
        node_name="concat",
        node_type="concat",
        node_metatype=PTCatMetatype,
        layer_attributes=PT2OpLayerAttributes(
            func=torch.concat,
            op_args=(
                [
                    TensorMeta(shape=(1,), dtype=torch.float),
                    TensorMeta(shape=(1,), dtype=torch.float),
                ]
            ),
            op_kwargs={},
            constant_port_ids=set(),
        ),
    )
    return graph


def _no_missed_input_edge_for_conv() -> PTNNCFGraph:
    graph = PTNNCFGraph()
    node_input = graph.add_nncf_node("input", "input", None, None)
    node_weight = graph.add_nncf_node("weight", "weight", None, None)
    node_conv = graph.add_nncf_node(
        node_name="conv",
        node_type="conv",
        node_metatype=PTConv2dMetatype,
        layer_attributes=PT2OpLayerAttributes(
            func=F.conv2d,
            op_args=(
                TensorMeta(shape=(1, 1, 1, 1), dtype=torch.float),
                TensorMeta(shape=(1, 1, 1, 1), dtype=torch.float),
            ),
            op_kwargs={},
            constant_port_ids=set([1]),
        ),
    )
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_conv.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_weight.node_id, node_conv.node_id, (1,), 1, 0, Dtype.FLOAT)
    return graph


@pytest.mark.parametrize(
    "graph_builder, ref",
    (
        (_missed_input_edge_for_conv, ["conv"]),
        (_missed_input_edge_for_concat, ["concat"]),
        (_no_missed_input_edge_for_conv, []),
    ),
)
def test_get_nodes_with_missed_input_edges(graph_builder: Callable[[], PTNNCFGraph], ref: list[str]):
    graph = graph_builder()
    ret = graph.get_nodes_with_missed_input_edges()
    ret_names = [node.node_name for node in ret]
    assert ret_names == ref
