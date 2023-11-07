# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nncf import NNCFConfig
from nncf.common.graph import NNCFGraphEdge
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import NNCFExtraConfigStruct
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch import create_compressed_model
from nncf.torch import nncf_model_input
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import EXTRA_STRUCTS_WITH_DATALOADERS
from nncf.torch.dynamic_graph.io_handling import ExactInputsInfo
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.io_handling import ModelInputInfo
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.dynamic_graph.trace_tensor import trace_tensors
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTGatherMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.operator_metatypes import PTSqueezeMetatype
from nncf.torch.graph.operator_metatypes import PTTransposeMetatype
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import NNCFNetworkMeta
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import get_basic_quantization_config
from tests.torch.test_get_modules_by_type import ModelForNameTest
from tests.torch.test_models.synthetic import ModelWithDummyParameter

TEST_TRACING_CONTEXT = "test"


def test_no_nncf_trace_context_manager():
    assert get_current_context() is None
    context = TracingContext()

    with context:
        assert get_current_context().is_tracing
        with no_nncf_trace():
            assert not get_current_context().is_tracing
            with no_nncf_trace():
                assert not get_current_context().is_tracing
            assert not get_current_context().is_tracing
        assert get_current_context().is_tracing
    assert get_current_context() is None


def test_ambiguous_function():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)])

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))

    mod = Model()

    tracer = GraphTracer(custom_forward_fn=create_dummy_forward_fn(FillerInputInfo([FillerInputElement([1, 1, 1, 1])])))
    graph = tracer.trace_graph(mod)

    unique_op_exec_contexts = set()

    for _, node in graph._nx_graph.nodes.items():
        node_op_address = node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR].op_address
        assert node_op_address not in unique_op_exec_contexts
        unique_op_exec_contexts.add(node_op_address)


def test_forward_trace_function():
    from nncf.torch.dynamic_graph.trace_functions import forward_trace_only
    from nncf.torch.dynamic_graph.trace_tensor import TensorMeta
    from nncf.torch.dynamic_graph.trace_tensor import TracedTensor

    shape1, shape2 = ([32, 1, 4, 8], [1, 8, 12, 16])
    meta1, meta2 = (TensorMeta(5, 1, shape1), TensorMeta(3, 8, shape2))
    input_tensor1 = TracedTensor.from_torch_tensor(torch.Tensor(size=shape1), meta1)
    input_tensor2 = TracedTensor.from_torch_tensor(torch.Tensor(size=shape2), meta2)

    # 1 -> 1
    output_tensor = forward_trace_only(torch.Tensor.view, input_tensor1, [-1])
    assert output_tensor.tensor_meta != input_tensor1.tensor_meta
    assert output_tensor.tensor_meta.shape == (1024,)

    # 1 -> N
    outputs = forward_trace_only(torch.Tensor.chunk, input_tensor1, 3)
    for out in outputs:
        assert out.tensor_meta == input_tensor1.tensor_meta

    # N -> N (2 -> 2)
    outputs = forward_trace_only(lambda x: x + [5], [input_tensor1, input_tensor2])
    assert outputs[0].tensor_meta == input_tensor1.tensor_meta
    assert outputs[1].tensor_meta == input_tensor2.tensor_meta

    # M -> N (2 -> 3)
    with pytest.raises(RuntimeError):
        outputs = forward_trace_only(lambda x: x + [torch.Tensor(shape2)], [input_tensor1, input_tensor2])

    # M -> N (2 -> 1)
    with pytest.raises(RuntimeError):
        outputs = forward_trace_only(lambda x: x[0], [input_tensor1, input_tensor2])


class ModelForTest(torch.nn.Module):
    IN_CHANNELS = 3
    OUT_CHANNELS = 10
    CONV1_OUT_CHANNELS = 15
    CONV2_IN_CHANNELS = CONV1_OUT_CHANNELS + IN_CHANNELS
    MAXPOOL_SIZE = 2

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(15)
        self.relu1 = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(self.CONV1_OUT_CHANNELS, self.IN_CHANNELS, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.CONV2_IN_CHANNELS, self.OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_prev = x
        x = F.max_pool2d(x, self.MAXPOOL_SIZE)
        x = self.convt1(x)
        x = torch.cat([x, x_prev], 1)
        x = self.conv2(x)
        return x

    @staticmethod
    def simple_wrap_fn(args, kwargs):
        arglist = list(args)
        arglist[0] = nncf_model_input(arglist[0])
        args = tuple(arglist)
        return args, kwargs

    @staticmethod
    def simple_user_dummy_forward(model):
        mock_tensor = torch.zeros(INPUT_SHAPES[0])
        args = (mock_tensor,)
        kwargs = {}
        args, kwargs = ModelForTest.simple_wrap_fn(args, kwargs)
        return wrap_nncf_model_outputs_with_objwalk(model(*args, **kwargs))


INPUT_SHAPES = [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 500, 500)]


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_activation_shape_tracing(input_shape: Tuple[int, ...]):
    model = ModelForTest()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)

    shape1 = (input_shape[0], ModelForTest.CONV1_OUT_CHANNELS, input_shape[2], input_shape[3])
    final_shape = (input_shape[0], ModelForTest.OUT_CHANNELS, input_shape[2], input_shape[3])
    ref_maxpool_out_edge_shapes = [
        (shape1[0], shape1[1], shape1[2] // ModelForTest.MAXPOOL_SIZE, shape1[3] // ModelForTest.MAXPOOL_SIZE)
    ]
    ref_cat_out_edge_shapes = [(input_shape[0], ModelForTest.CONV2_IN_CHANNELS, input_shape[2], input_shape[3])]
    ref_node_ids_and_io_edge_shapes = [
        (f"0 /{MODEL_INPUT_OP_NAME}_0", [], [input_shape]),
        ("1 ModelForTest/Conv2d[conv1]/conv2d_0", [input_shape], [shape1]),
        ("2 ModelForTest/BatchNorm2d[bn1]/batch_norm_0", [shape1], [shape1]),
        ("3 ModelForTest/ReLU[relu1]/relu_0", [shape1], [shape1, shape1]),
        ("4 ModelForTest/max_pool2d_0", [shape1], ref_maxpool_out_edge_shapes),
        ("5 ModelForTest/ConvTranspose2d[convt1]/conv_transpose2d_0", ref_maxpool_out_edge_shapes, [input_shape]),
        ("6 ModelForTest/cat_0", [shape1, input_shape], ref_cat_out_edge_shapes),
        ("7 ModelForTest/Conv2d[conv2]/conv2d_0", ref_cat_out_edge_shapes, [final_shape]),
        (f"8 /{MODEL_OUTPUT_OP_NAME}_0", [final_shape], []),
    ]
    for node_id, ref_input_shapes, ref_output_shapes in ref_node_ids_and_io_edge_shapes:
        input_edges = graph.get_nncf_graph_pattern_io(
            [
                node_id,
            ]
        ).input_edges
        output_edges = graph.get_nncf_graph_pattern_io(
            [
                node_id,
            ]
        ).output_edges
        input_tensor_shapes = [x.tensor_shape for x in input_edges]
        output_tensor_shapes = [x.tensor_shape for x in output_edges]
        assert input_tensor_shapes == ref_input_shapes, "Failed for node ID: {}".format(node_id)
        assert output_tensor_shapes == ref_output_shapes, "Failed for node ID: {}".format(node_id)


class ModelForTestWithReshapeFlattenAndConcat(ModelForTest):
    def forward(self, x):
        y = super().forward(x)
        size = y.size()
        y = y.view(size + (1, 1))

        y_copy = torch.ones_like(y)
        y = torch.stack([y, y_copy])

        y_copy = torch.ones_like(y)
        y = torch.cat([y, y_copy], -1)

        y = torch.flatten(y)
        _ = y.view(-1)

        y_copy = torch.ones_like(y)
        y = torch.stack([y, y_copy])

        y_copy = torch.ones_like(y)
        y = torch.cat([y, y_copy], -1)
        return y


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_concat_attributes_saved_during_graph_building(input_shape):
    model = ModelForTestWithReshapeFlattenAndConcat()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo([FillerInputElement(input_shape)]),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    cat_nodes_with_attributes = {
        "ModelForTestWithReshapeFlattenAndConcat/cat_0": {"axis": 1},
        "ModelForTestWithReshapeFlattenAndConcat/cat_1": {"axis": 6},
        "ModelForTestWithReshapeFlattenAndConcat/cat_2": {"axis": 1},
        "ModelForTestWithReshapeFlattenAndConcat/stack_0": None,
        "ModelForTestWithReshapeFlattenAndConcat/stack_1": None,
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTCatMetatype:
            assert node.node_name in cat_nodes_with_attributes
            if isinstance(node.layer_attributes, MultipleInputLayerAttributes):
                assert node.layer_attributes.axis == cat_nodes_with_attributes[node.node_name]["axis"]
            else:
                assert node.layer_attributes is None
                assert cat_nodes_with_attributes[node.node_name] is None


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_reshape_attributes_saved_during_graph_building(input_shape):
    model = ModelForTestWithReshapeFlattenAndConcat()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    reshape_nodes_with_attributes = {
        "ModelForTestWithReshapeFlattenAndConcat/view_0": {
            "input_shape": (input_shape[0], ModelForTest.OUT_CHANNELS, input_shape[2], input_shape[3]),
            "output_shape": (input_shape[0], ModelForTest.OUT_CHANNELS, input_shape[2], input_shape[3], 1, 1),
        },
        "ModelForTestWithReshapeFlattenAndConcat/flatten_0": {
            "input_shape": (2, input_shape[0], ModelForTest.OUT_CHANNELS, input_shape[2], input_shape[3], 1, 2),
            "output_shape": (input_shape[0] * ModelForTest.OUT_CHANNELS * input_shape[2] * input_shape[3] * 4,),
        },
        "ModelForTestWithReshapeFlattenAndConcat/view_1": None,
    }

    for node in graph.get_all_nodes():
        if node.metatype in [PTReshapeMetatype, PTSqueezeMetatype]:
            assert node.node_name in reshape_nodes_with_attributes
            if isinstance(node.layer_attributes, ReshapeLayerAttributes):
                ref_attrs = reshape_nodes_with_attributes[node.node_name]
                assert node.layer_attributes.input_shape == ref_attrs["input_shape"]
                assert node.layer_attributes.output_shape == ref_attrs["output_shape"]
            else:
                assert node.layer_attributes is None
                assert reshape_nodes_with_attributes[node.node_name] is None


class ModelWithPermute(nn.Module):
    def forward(self, x: torch.Tensor):
        # x.shape == [1, 10, 20, 10]
        # without kwargs
        x = x.transpose(1, 3)
        x = x.permute(3, 2, 1, 0)
        # with kwargs
        x = x.transpose(1, dim1=3)
        x = x.transpose(dim0=1, dim1=3)
        x = x.permute(dims=[3, 2, 1, 0])
        return x


transpose_input_shapes = [(1, 10, 20, 10), (10, 10, 10, 10)]


@pytest.mark.parametrize("input_shape", transpose_input_shapes)
def test_permute_attributes_saved_during_graph_building(input_shape):
    model = ModelWithPermute()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    transpose_nodes_with_attributes = {
        "ModelWithPermute/transpose_0": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/transpose_1": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/transpose_2": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/permute_0": PermuteLayerAttributes((3, 2, 1, 0)),
        "ModelWithPermute/permute_1": PermuteLayerAttributes([3, 2, 1, 0]),
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTTransposeMetatype:
            assert node.node_name in transpose_nodes_with_attributes
            if isinstance(node.layer_attributes, (TransposeLayerAttributes, PermuteLayerAttributes)):
                ref_attrs = transpose_nodes_with_attributes[node.node_name]
                assert node.layer_attributes == ref_attrs
            else:
                assert node.layer_attributes is None
                assert transpose_nodes_with_attributes[node.node_name] is None


class ModelForTestWithSplit(ModelForTest):
    def __init__(self, input_shape):
        super().__init__()
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(input_shape[0], 1, kernel_size=1, padding=0)

    def forward(self, x):
        y = super().forward(x)
        y1, y2 = torch.chunk(y, chunks=2, dim=1)

        y1 = self.conv3(y1)
        y2 = self.conv3(y2)
        y = torch.cat([y1, y2], axis=1)

        y_unbinded = torch.unbind(y, dim=1)
        unbinded_processed = list(y_unbinded)
        unbinded_processed[0] = self.conv4(y_unbinded[0])
        y = torch.cat(unbinded_processed, axis=0)
        return y


@pytest.mark.parametrize("input_shape", INPUT_SHAPES)
def test_split_attributes(input_shape):
    model = ModelForTestWithSplit(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )

    graph = graph_builder.build_graph(model)
    chunk_nodes_with_attributes = {
        "ModelForTestWithSplit/chunk_0": {"chunks": 2, "axis": 1},
        "ModelForTestWithSplit/unbind_0": {"chunks": 20, "axis": 1},
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTSplitMetatype:
            assert node.node_name in chunk_nodes_with_attributes
            if isinstance(node.layer_attributes, MultipleOutputLayerAttributes):
                ref_attrs = chunk_nodes_with_attributes[node.node_name]
                assert node.layer_attributes.chunks == ref_attrs["chunks"]
                assert node.layer_attributes.axis == ref_attrs["axis"]
            else:
                assert node.layer_attributes is None
                assert chunk_nodes_with_attributes[node.node_name] is None


class SplitByGetItemModel(ModelWithDummyParameter):
    def forward(self, x):
        return x[0:1], x[(0, 1)], x[2]


@pytest.mark.parametrize("input_shape", [(3, 2)])
def test_getitem_attributes(input_shape):
    model = SplitByGetItemModel()
    custom_forward_fn = create_dummy_forward_fn(
        FillerInputInfo(
            [
                FillerInputElement(input_shape),
            ]
        ),
        with_input_tracing=True,
        with_output_tracing=True,
    )
    graph_builder = GraphBuilder(custom_forward_fn)
    graph = graph_builder.build_graph(model)
    getitem_nodes_with_attributes = {
        "SplitByGetItemModel/__getitem___0": slice(0, 1, None),
        "SplitByGetItemModel/__getitem___1": (0, 1),
        "SplitByGetItemModel/__getitem___2": 2,
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTGatherMetatype:
            assert node.node_name in getitem_nodes_with_attributes
            if isinstance(node.layer_attributes, GetItemLayerAttributes):
                ref_key = getitem_nodes_with_attributes[node.node_name]
                assert node.layer_attributes.key == ref_key
            else:
                assert node.layer_attributes is None
                assert getitem_nodes_with_attributes[node.node_name] is None


class ParallelEdgesModel(nn.Module):
    def forward(self, x):
        mm_res = torch.mm(x, x)
        return mm_res, x + mm_res


def test_parallel_edges_in_nncf_graph():
    def _get_default_nncf_graph_edge(from_node, to_node, input_port_id, output_port_id):
        return NNCFGraphEdge(
            from_node,
            to_node,
            input_port_id=input_port_id,
            output_port_id=output_port_id,
            tensor_shape=(3, 3),
            dtype=Dtype.FLOAT,
            parallel_input_port_ids=[],
        )

    input_shape = (3, 3)
    model = ParallelEdgesModel()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )

    nncf_graph = graph_builder.build_graph(model)

    input_node = nncf_graph.get_node_by_name("/nncf_model_input_0")
    mm_node = nncf_graph.get_node_by_name("ParallelEdgesModel/mm_0")
    ref_input_edges = {
        _get_default_nncf_graph_edge(input_node, mm_node, input_port_id=0, output_port_id=0),
        _get_default_nncf_graph_edge(input_node, mm_node, input_port_id=1, output_port_id=0),
    }
    mm_node_input_edges = nncf_graph.get_input_edges(mm_node)
    assert set(mm_node_input_edges) == ref_input_edges
    ref_output_edges = ref_input_edges.copy()

    add_node = nncf_graph.get_node_by_name("ParallelEdgesModel/__add___0")
    ref_output_edges.add(_get_default_nncf_graph_edge(input_node, add_node, input_port_id=0, output_port_id=0))
    input_node_output_edges = nncf_graph.get_output_edges(input_node)
    assert set(input_node_output_edges) == ref_output_edges


class MockModel(torch.nn.Module):
    def __init__(self, stub_forward):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones([1]))
        self.stub_forward = stub_forward

    def forward(self, *args, **kwargs):
        return self.stub_forward(*args, **kwargs)


class RandomRefTensor:
    def __init__(self, shape: List[int]):
        self.tensor = torch.rand(shape)


def check_arg(test_arg: torch.Tensor, ref_arg: Union[torch.Tensor, RandomRefTensor]):
    if isinstance(ref_arg, RandomRefTensor):
        assert test_arg.shape == ref_arg.tensor.shape
        assert not torch.allclose(test_arg, torch.ones_like(test_arg))
        assert not torch.allclose(test_arg, torch.zeros_like(test_arg))
    else:
        assert torch.allclose(test_arg, ref_arg)


class MockInputInfo(ModelInputInfo):
    MOCK_ARGS = (torch.Tensor([42.0]),)
    MOCK_KWARGS = {"foo": torch.ones([1, 3])}

    def get_forward_inputs(self, device: str = None) -> Tuple[Tuple, Dict]:
        return MockInputInfo.MOCK_ARGS, MockInputInfo.MOCK_KWARGS


@pytest.fixture(scope="function")
def mock_model_with_stub_forward(mocker) -> MockModel:
    stub_fn = mocker.stub()
    mock_model = MockModel(stub_fn)
    return mock_model


def test_input_info_args_are_passed_into_forward(mock_model_with_stub_forward: MockModel):
    stub_fn = mock_model_with_stub_forward.stub_forward

    _ = NNCFNetwork(mock_model_with_stub_forward, input_info=MockInputInfo())
    forward_call_args = stub_fn.call_args[0]
    forward_call_kwargs = stub_fn.call_args[1]

    ref_args, ref_kwargs = MockInputInfo.MOCK_ARGS, MockInputInfo.MOCK_KWARGS

    assert len(forward_call_args) == len(ref_args)
    assert len(forward_call_kwargs) == len(ref_kwargs)
    assert set(forward_call_kwargs.keys()) == set(ref_kwargs.keys())

    for idx, arg in enumerate(forward_call_args):
        check_arg(arg, ref_args[idx])

    for keyword, arg in forward_call_kwargs.items():
        check_arg(arg, ref_kwargs[keyword])


@dataclass
class FillerInputInfoGenerationTestStruct:
    config_input_info_subdict: Union[List[Dict], Dict]
    ref_args: Tuple[torch.Tensor, ...]
    ref_kwargs: Dict[str, torch.Tensor]


TEST_KEYWORD_1 = "keyword1"
TEST_KEYWORD_2 = "keyword2"

FILLER_GEN_TEST_STRUCTS = [
    FillerInputInfoGenerationTestStruct(
        config_input_info_subdict={"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
        ref_args=(torch.zeros([2, 3, 300, 300]),),
        ref_kwargs={},
    ),
    FillerInputInfoGenerationTestStruct(
        config_input_info_subdict=[
            {"sample_size": [1, 128], "type": "long", "filler": "ones"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones"},
            {"sample_size": [1, 128], "type": "long", "filler": "zeros"},
        ],
        ref_args=(
            torch.ones([1, 128], dtype=torch.long),
            torch.ones([1, 128], dtype=torch.long),
            torch.zeros([1, 128], dtype=torch.long),
        ),
        ref_kwargs={},
    ),
    FillerInputInfoGenerationTestStruct(
        config_input_info_subdict=[
            {"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones", "keyword": TEST_KEYWORD_1},
        ],
        ref_args=(torch.zeros([2, 3, 300, 300]),),
        ref_kwargs={TEST_KEYWORD_1: torch.ones([1, 128], dtype=torch.long)},
    ),
    FillerInputInfoGenerationTestStruct(
        config_input_info_subdict=[
            {"sample_size": [8, 7], "type": "float", "filler": "random", "keyword": TEST_KEYWORD_1},
            {"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones", "keyword": TEST_KEYWORD_2},
        ],
        ref_args=(torch.zeros([2, 3, 300, 300]),),
        ref_kwargs={TEST_KEYWORD_1: RandomRefTensor([8, 7]), TEST_KEYWORD_2: torch.ones([1, 128], dtype=torch.long)},
    ),
]


@pytest.mark.parametrize("filler_gen_test_struct", FILLER_GEN_TEST_STRUCTS)
def test_filler_input_info_arg_generation(filler_gen_test_struct: FillerInputInfoGenerationTestStruct):
    filler_input_info = FillerInputInfo.from_nncf_config(
        NNCFConfig.from_dict({"input_info": filler_gen_test_struct.config_input_info_subdict})
    )
    test_args, test_kwargs = filler_input_info.get_forward_inputs()

    for test_arg, ref_arg in zip(test_args, filler_gen_test_struct.ref_args):
        check_arg(test_arg, ref_arg)

    for test_kw_and_arg, ref_kw_and_arg in zip(test_kwargs.items(), filler_gen_test_struct.ref_kwargs.items()):
        test_kw, test_kwarg = test_kw_and_arg
        ref_kw, ref_kwarg = ref_kw_and_arg
        assert test_kw == ref_kw
        check_arg(test_kwarg, ref_kwarg)


class MockInitDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        return dataloader_output[0], dataloader_output[1]

    def get_target(self, dataloader_output: Any) -> Any:
        return torch.empty([1])


class MockDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._length = 2

    def __getitem__(self, idx):
        if idx >= self._length:
            raise StopIteration
        return MockInputInfo.MOCK_ARGS, MockInputInfo.MOCK_KWARGS

    def __len__(self):
        return self._length


STRUCTS_FOR_TEST = [
    QuantizationRangeInitArgs(data_loader=MockInitDataLoader(torch.utils.data.DataLoader(MockDataset()))),
    BNAdaptationInitArgs(data_loader=MockInitDataLoader(torch.utils.data.DataLoader(MockDataset()))),
]


@pytest.mark.parametrize("extra_struct_for_test", STRUCTS_FOR_TEST)
def test_compressed_model_creation_can_build_exact_input_infos_from_dataloader_in_config(
    extra_struct_for_test: NNCFExtraConfigStruct, mock_model_with_stub_forward: MockModel, mocker
):
    checked_types_set = {type(x) for x in STRUCTS_FOR_TEST}
    assert checked_types_set == set(
        EXTRA_STRUCTS_WITH_DATALOADERS
    )  # all future structs with suitable dataloaders must be tested
    config = NNCFConfig()
    config.register_extra_structs([extra_struct_for_test])
    nncf_network_init_spy = mocker.spy(NNCFNetworkMeta, "__call__")  # sic!

    _ = create_compressed_model(mock_model_with_stub_forward, config)
    input_info_received_by_nncf_network_init = nncf_network_init_spy.call_args.kwargs["input_info"]  # input_info
    assert isinstance(input_info_received_by_nncf_network_init, ExactInputsInfo)
    test_args, test_kwargs = input_info_received_by_nncf_network_init.get_forward_inputs()

    for idx, arg in enumerate(test_args):
        check_arg(arg, MockInputInfo.MOCK_ARGS[idx])

    for keyword, arg in test_kwargs.items():
        check_arg(arg, MockInputInfo.MOCK_KWARGS[keyword])


def create_model_and_control_with_defaults():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(INPUT_SHAPES[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return compressed_model, compression_ctrl


def create_model_with_user_dummy():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(INPUT_SHAPES[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model,
        config,
        dummy_forward_fn=ModelForTest.simple_user_dummy_forward,
        wrap_inputs_fn=ModelForTest.simple_wrap_fn,
    )
    return compressed_model, compression_ctrl


def create_model_with_user_wrap_inputs_fn():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(INPUT_SHAPES[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model,
        config,
        dummy_forward_fn=ModelForTest.simple_user_dummy_forward,
        wrap_inputs_fn=ModelForTest.simple_wrap_fn,
    )
    return compressed_model, compression_ctrl


class TestGraphStability:
    MODEL_CREATORS_AND_IDS = [
        (create_model_and_control_with_defaults, "default"),
        (create_model_with_user_dummy, "user_dummy"),
        (create_model_with_user_wrap_inputs_fn, "user_wrap_inputs_fn"),
    ]

    @pytest.fixture(params=[x[0] for x in MODEL_CREATORS_AND_IDS], ids=[x[1] for x in MODEL_CREATORS_AND_IDS])
    def model_and_ctrl_creator(self, request):
        return request.param

    def test_dynamic_graph_does_not_inflate_during_multiple_forwards(self, model_and_ctrl_creator):
        compressed_model, _ = model_and_ctrl_creator()
        input_tensor = torch.zeros(INPUT_SHAPES[0])
        ref_graph = deepcopy(compressed_model.nncf.get_dynamic_graph())
        for _ in range(0, 10):
            _ = compressed_model(input_tensor)
            curr_graph = compressed_model.nncf.get_dynamic_graph()
            assert curr_graph == ref_graph

    def test_dynamic_graph_is_the_same_after_export(self, model_and_ctrl_creator, tmp_path):
        compressed_model, ctrl = model_and_ctrl_creator()
        ref_graph = deepcopy(compressed_model.nncf.get_dynamic_graph())
        ctrl.export_model("tmp.onnx")
        curr_graph = compressed_model.nncf.get_dynamic_graph()
        assert curr_graph == ref_graph

    def test_dummy_forwards_do_not_inflate_dynamic_graph(self, model_and_ctrl_creator):
        compressed_model, _ = model_and_ctrl_creator()
        ref_graph = deepcopy(compressed_model.nncf.get_dynamic_graph())
        compressed_model.nncf.do_dummy_forward()
        curr_graph = deepcopy(compressed_model.nncf.get_dynamic_graph())
        assert curr_graph == ref_graph

    def test_compressed_graph_with_user_wrap_fn(self):
        # Create a model with a dummy forward analogous to
        # the default dummy forward, compare original and compressed model graphs afterwards

        comp_model_wo_wrap, _ = create_model_and_control_with_defaults()
        comp_model_w_wrap, _ = create_model_with_user_wrap_inputs_fn()

        ref_original_graph = comp_model_wo_wrap.nncf.get_graph()
        ref_compressed_graph = comp_model_wo_wrap.nncf.get_graph()

        original_graph_with_wrap = comp_model_w_wrap.nncf.get_graph()
        compressed_graph_with_wrap = comp_model_w_wrap.nncf.get_graph()

        assert ref_original_graph == original_graph_with_wrap
        assert ref_compressed_graph == compressed_graph_with_wrap

    def test_compressed_graph_with_user_dummy_forward(self):
        # Create a model with a dummy forward analogous to
        # the default dummy forward, compare original and compressed model graphs afterwards

        comp_model_wo_dummy, _ = create_model_and_control_with_defaults()
        comp_model_w_dummy, _ = create_model_with_user_dummy()

        ref_original_graph = comp_model_wo_dummy.nncf.get_graph()
        ref_compressed_graph = comp_model_wo_dummy.nncf.get_graph()

        original_graph_with_dummy = comp_model_w_dummy.nncf.get_graph()
        compressed_graph_with_dummy = comp_model_w_dummy.nncf.get_graph()

        assert ref_original_graph == original_graph_with_dummy
        assert ref_compressed_graph == compressed_graph_with_dummy


def test_nncf_graph_auxiliary_node_structure():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(INPUT_SHAPES[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    nncf_graph = compressed_model.nncf.get_graph()

    input_nodes = nncf_graph.get_input_nodes()
    output_nodes = nncf_graph.get_output_nodes()

    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

    assert input_nodes[0].node_type == NNCFGraphNodeType.INPUT_NODE
    assert output_nodes[0].node_type == NNCFGraphNodeType.OUTPUT_NODE


def test_get_all_nodes():
    model = ModelForNameTest()
    ref_list = [
        "ModelForNameTest/Conv2d[conv1]/conv2d_0",
        "ModelForNameTest/BatchNorm2d[bn1]/batch_norm_0",
        "ModelForNameTest/ReLU/relu_0",
        "ModelForNameTest/relu_0",
        "ModelForNameTest/BatchNorm2d[bn2]/batch_norm_0",
        "ModelForNameTest/Sequential[layer2]/Sequential[layer1]/Conv2d[conv01]/conv2d_0",
        "ModelForNameTest/Sequential[layer2]/Sequential[layer1]/BatchNorm2d[norm01]/batch_norm_0",
        "ModelForNameTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]/relu_0",
        "ModelForNameTest/Sequential[layer2]/Sequential[layer1]/MaxPool2d[pool01]/max_pool2d_0",
        "ModelForNameTest/Sequential[layer2]/Conv2d[conv02]/conv2d_0",
        "ModelForNameTest/Sequential[layer2]/ReLU[relu02]/relu_0",
        "ModelForNameTest/Sequential[layer2]/BatchNorm2d[norm02]/batch_norm_0",
        "ModelForNameTest/Sequential[layer2]/MaxPool2d[pool02]/max_pool2d_0",
        "ModelForNameTest/AvgPool2d[avgpool]/avg_pool2d_0",
    ]

    builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement((1, 1, 4, 4)),
                ]
            )
        )
    )
    graph = builder.build_graph(model)
    test_list = [node_name.split(" ", 1)[1] for node_name in graph.get_all_node_keys()]
    assert ref_list == test_list


class ModelWithIntegerPaths(torch.nn.Module):
    INPUT_SHAPE = [2, 2, 2, 2]

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1)
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        sz = torch.tensor(x.shape).to(x.device)
        sz_tensor = torch.cat([sz])
        idx_tensor = sz_tensor // sz_tensor
        single_idx = idx_tensor[0]
        y = x[single_idx][single_idx] * torch.ones([1, 1]).to(x.device)
        z = self.linear(y)
        return z


def test_integer_path_marking():
    input_infos = FillerInputInfo(
        [
            FillerInputElement(ModelWithIntegerPaths.INPUT_SHAPE),
        ]
    )
    builder = GraphBuilder(create_dummy_forward_fn(input_infos))
    nncf_graph = builder.build_graph(ModelWithIntegerPaths())
    edges = list(nncf_graph.get_all_edges())
    num_integer_edges = sum([1 for edge in edges if edge.dtype is Dtype.INTEGER])
    # cat -> __floordiv__,  __floordiv__ -> __getitem__0 (to get single_idx),
    # __getitem__0 -> __getitem__1 (first indexing by tensor), __getitem__0 -> __getitem__2 (second indexing by tensor)
    assert num_integer_edges == 4


def test_trace_output_with_no_tensors():
    output = None
    trace_tensors(output, MagicMock())


class ModelWithRepeatInputs(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        return torch.stack([x, y, x, y])


def test_dynamic_graph_assigns_contiguous_input_ports_for_edges_with_multiplicity():
    input_info = FillerInputInfo([FillerInputElement([1, 3, 3, 3])])
    tracer = GraphTracer(create_dummy_forward_fn(input_info, with_input_tracing=True, with_output_tracing=True))
    dynamic_graph = tracer.trace_graph(ModelWithRepeatInputs())
    stack_in_edges = [e for e in dynamic_graph.get_all_edges() if e.to_node_id == 2]  # node id 2 == torch.stack
    all_input_port_ids = set()
    for edge in stack_in_edges:
        all_input_port_ids.add(edge.input_port_id)
        all_input_port_ids.update(edge.parallel_input_port_ids)
    assert all_input_port_ids == {0, 1, 2, 3}
