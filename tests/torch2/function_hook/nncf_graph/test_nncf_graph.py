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

from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Union

import networkx as nx
import pytest
import torch
import torchvision.models as models

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import NodeType
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import convert_to_nncf_graph
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_dtype
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_name_of_node
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_node_type
from nncf.experimental.torch2.function_hook.wrapper import wrap_model
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch2.function_hook import helpers
from tests.torch2.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "nncf_graph"


def get_reference_graph(graph: NNCFGraph) -> nx.DiGraph:
    out_graph = nx.DiGraph()
    for node in sorted(graph.get_all_nodes(), key=lambda x: x.node_id):
        attrs_node = {
            "id": node.node_id,
            "type": node.node_type,
            "metatype": node.metatype.__name__,
        }
        out_graph.add_node(node.node_name, **attrs_node)

    for edge in graph.get_all_edges():
        attrs_edge = {"dtype": edge.dtype.value, "shape": edge.tensor_shape}
        if edge.parallel_input_port_ids:
            attrs_edge["parallel_input_port_ids"] = edge.parallel_input_port_ids

        out_graph.add_edge(edge.from_node.node_name, edge.to_node.node_name, **attrs_edge)
    return out_graph


@pytest.mark.parametrize(
    "node_type, meta, ref",
    [
        [NodeType.input, InOutMeta(torch.float32, (1), "input"), "nncf_model_input"],
        [NodeType.output, InOutMeta(torch.float32, (1), "output"), "nncf_model_output"],
        [NodeType.output, FunctionMeta("op", "fn_name_ref", [], {}), "fn_name_ref"],
        [NodeType.output, ConstMeta(torch.float32, (1), "model.bias"), "nncf_model_const"],
    ],
)
def test_get_node_type(node_type: NodeType, meta: Union[ConstMeta, FunctionMeta, InOutMeta], ref: str):
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
def test_get_name_of_node(meta: Union[InOutMeta, FunctionMeta, ConstMeta], ref: str):
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
    graph = get_reference_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)

    ref_file = REF_DIR / "convert_to_nncf_graph.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


def test_convert_to_nncf_graph_multi_edges(regen_ref_data: bool):
    model = helpers.ModelMultiEdge()
    model = wrap_model(model)
    nx_graph = build_graph(model, torch.ones(1, 1))
    nncf_graph = convert_to_nncf_graph(nx_graph)
    graph = get_reference_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / "convert_to_nncf_graph_multi_edges.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


@dataclass
class ModelDesc:
    model_name: str
    model_builder: callable
    inputs_info: Union[List[List[int]], Tuple[List[int], ...]]

    def __str__(self):
        return self.model_name


TEST_MODELS_DESC = [
    ModelDesc("convnext_small", partial(models.convnext_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("densenet121", partial(models.densenet121, weights=None), [1, 3, 64, 64]),
    ModelDesc("efficientnet_b0", partial(models.efficientnet_b0, weights=None), [1, 3, 64, 64]),
    ModelDesc("inception_v3", partial(models.inception_v3, weights=None), [1, 3, 300, 300]),
    ModelDesc("mobilenet_v2", partial(models.mobilenet_v2, weights=None), [1, 3, 64, 64]),
    ModelDesc("mobilenet_v3_small", partial(models.mobilenet_v3_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnet18", partial(models.resnet18, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnext50_32x4d", partial(models.resnext50_32x4d, weights=None), [1, 3, 64, 64]),
    ModelDesc("shufflenet_v2_x0_5", partial(models.shufflenet_v2_x0_5, weights=None), [1, 3, 224, 224]),
    ModelDesc("squeezenet1_0", partial(models.squeezenet1_0, weights=None), [1, 3, 64, 64]),
    ModelDesc("swin_v2_b", partial(models.swin_v2_b, weights=None), [1, 3, 64, 64]),
    ModelDesc("vgg16", partial(models.vgg16, weights=None), [1, 3, 32, 32]),
]


@pytest.mark.parametrize("desc", TEST_MODELS_DESC, ids=str)
def test_model_graph(desc: ModelDesc, regen_ref_data: bool):
    model: torch.nn.Module = desc.model_builder()
    model = model.eval()
    inputs = [torch.randn(desc.inputs_info)]
    model = wrap_model(model)
    nncf_graph = build_nncf_graph(model, *inputs)
    graph = get_reference_graph(nncf_graph)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / f"model_graph_{desc}.dot"
    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)
