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
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nncf.common.graph import Dtype
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.torch import nncf_model_input
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.dynamic_graph.trace_tensor import trace_tensors
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTGatherMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.operator_metatypes import PTTransposeMetatype
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
    input_info = ModelInputInfo([1, 1, 1, 1])

    tracer = GraphTracer(
        custom_forward_fn=create_dummy_forward_fn(
            [
                input_info,
            ]
        )
    )
    graph = tracer.trace_graph(mod)

    unique_op_exec_contexts = set()
    # pylint:disable=protected-access
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
        mock_tensor = torch.zeros(input_shapes[0])
        args = (mock_tensor,)
        kwargs = {}
        args, kwargs = ModelForTest.simple_wrap_fn(args, kwargs)
        return wrap_nncf_model_outputs_with_objwalk(model(*args, **kwargs))


input_shapes = [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 500, 500)]


@pytest.mark.parametrize("input_shape", input_shapes)
def test_activation_shape_tracing(input_shape: Tuple):
    model = ModelForTest()
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            [
                input_info,
            ],
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
        # pylint:disable=protected-access
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


@pytest.mark.parametrize("input_shape", input_shapes)
def test_concat_attributes_saved_during_graph_building(input_shape):
    model = ModelForTestWithReshapeFlattenAndConcat()
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            [
                input_info,
            ],
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


@pytest.mark.parametrize("input_shape", input_shapes)
def test_reshape_attributes_saved_during_graph_building(input_shape):
    model = ModelForTestWithReshapeFlattenAndConcat()
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            [
                input_info,
            ],
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
        if node.metatype is PTReshapeMetatype:
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
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            [
                input_info,
            ],
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
        "ModelWithPermute/permute_1": PermuteLayerAttributes((3, 2, 1, 0)),
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


@pytest.mark.parametrize("input_shape", input_shapes)
def test_split_attributes(input_shape):
    model = ModelForTestWithSplit(input_shape)
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            [
                input_info,
            ],
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )

    graph = graph_builder.build_graph(model)
    chunk_nodes_with_attributes = {
        "ModelForTestWithSplit/chunk_0": {"chunks": 2, "axis": 1},
        "ModelForTestWithSplit/unbind_0": {"chunks": 2, "axis": 1}
        # TODO: fix NNCFGraph tracing so valid reference below
        # will be generated by NNCF
        #'ModelForTestWithSplit/unbind_0': {'chunks': 20, 'axis': 1}
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
    input_info = ModelInputInfo(input_shape)
    custom_forward_fn = create_dummy_forward_fn(
        [
            input_info,
        ],
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


TEST_KEYWORD_1 = "keyword1"
TEST_KEYWORD_2 = "keyword2"
INPUT_INFO_CONFIG_VS_FORWARD_ARGS = [
    (
        {"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
        [ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS)],
    ),
    (
        [
            {"sample_size": [1, 128], "type": "long", "filler": "ones"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones"},
            {"sample_size": [1, 128], "type": "long", "filler": "zeros"},
        ],
        [
            ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES),
            ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES),
            ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES),
        ],
    ),
    (
        [
            {"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones", "keyword": TEST_KEYWORD_1},
        ],
        [
            ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS),
            ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_1),
        ],
    ),
    (
        [
            {"sample_size": [8, 7], "type": "float", "filler": "random", "keyword": TEST_KEYWORD_1},
            {"sample_size": [2, 3, 300, 300], "type": "float", "filler": "zeros"},
            {"sample_size": [1, 128], "type": "long", "filler": "ones", "keyword": TEST_KEYWORD_2},
        ],
        [
            ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS),
            ModelInputInfo([8, 7], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_1),
            ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_2),
        ],
    ),
]


class MockModel(torch.nn.Module):
    def __init__(self, stub_forward):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones([1]))
        self.stub_forward = stub_forward

    def forward(self, *args, **kwargs):
        return self.stub_forward(*args, **kwargs)


@pytest.fixture(params=INPUT_INFO_CONFIG_VS_FORWARD_ARGS, name="input_info_test_struct")
def input_info_test_struct_(request):
    return request.param


def test_input_info_specification_from_config(mocker, input_info_test_struct):
    stub_fn = mocker.stub()
    mock_model = MockModel(stub_fn)
    config = get_basic_quantization_config("symmetric")
    input_info_config_entry = input_info_test_struct[0]
    target_argument_info = input_info_test_struct[1]  # type: List[ModelInputInfo]
    config["input_info"] = input_info_config_entry
    register_bn_adaptation_init_args(config)

    _, _ = create_compressed_model_and_algo_for_test(mock_model, config)
    forward_call_args = stub_fn.call_args[0]
    forward_call_kwargs = stub_fn.call_args[1]

    ref_args_info = list(filter(lambda x: x.keyword is None, target_argument_info))
    ref_kw_vs_arg_info = {x.keyword: x for x in target_argument_info if x.keyword is not None}

    def check_arg(arg: torch.Tensor, ref_arg_info: ModelInputInfo):
        assert list(arg.shape) == ref_arg_info.shape
        assert arg.dtype == ref_arg_info.type

    assert len(forward_call_args) == len(ref_args_info)
    assert len(forward_call_kwargs) == len(ref_kw_vs_arg_info)
    assert set(forward_call_kwargs.keys()) == set(ref_kw_vs_arg_info.keys())

    for idx, arg in enumerate(forward_call_args):
        check_arg(arg, ref_args_info[idx])

    for keyword, arg in forward_call_kwargs.items():
        check_arg(arg, ref_kw_vs_arg_info[keyword])


def create_model_and_control_with_defaults():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return compressed_model, compression_ctrl


def create_model_with_user_dummy():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
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
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
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
        input_tensor = torch.zeros(input_shapes[0])
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
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
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
            [
                ModelInputInfo((1, 1, 4, 4)),
            ]
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
    input_infos = [
        ModelInputInfo(ModelWithIntegerPaths.INPUT_SHAPE),
    ]
    builder = GraphBuilder(create_dummy_forward_fn(input_infos))
    nncf_graph = builder.build_graph(ModelWithIntegerPaths(), input_infos=input_infos)
    edges = list(nncf_graph.get_all_edges())
    num_integer_edges = sum([1 for edge in edges if edge.dtype is Dtype.INTEGER])
    # cat -> __floordiv__,  __floordiv__ -> __getitem__0 (to get single_idx),
    # __getitem__0 -> __getitem__1 (first indexing by tensor), __getitem__0 -> __getitem__2 (second indexing by tensor)
    assert num_integer_edges == 4


def test_trace_output_with_no_tensors():
    output = None
    trace_tensors(output, MagicMock())
