"""
 Copyright (c) 2019-2020 Intel Corporation
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
from copy import deepcopy

from nncf.common.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph import MODEL_OUTPUT_OP_NAME
from nncf.common.graph import NNCFGraphNodeType
from typing import List
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncf.torch import nncf_model_input
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.context import get_current_context
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.graph.graph_builder import GraphBuilder
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import get_basic_quantization_config
from tests.torch.test_get_modules_by_type import ModelForNameTest

TEST_TRACING_CONTEXT = 'test'


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
            self.layers = nn.ModuleList([
                nn.Conv2d(1, 1, 1),
                nn.Conv2d(1, 1, 1)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))

    mod = Model()
    input_info = ModelInputInfo([1, 1, 1, 1])

    tracer = GraphTracer(custom_forward_fn=create_dummy_forward_fn([input_info, ]))
    graph = tracer.trace_graph(mod)

    unique_op_exec_contexts = set()
    # pylint:disable=protected-access
    for _, node in graph._nx_graph.nodes.items():
        node_op_address = node[DynamicGraph.OP_EXEC_CONTEXT_NODE_ATTR].op_address
        assert node_op_address not in unique_op_exec_contexts
        unique_op_exec_contexts.add(node_op_address)


def test_forward_trace_functor():
    from nncf.torch.dynamic_graph.patch_pytorch import ForwardTraceOnly
    from nncf.torch.dynamic_graph.trace_tensor import TracedTensor, TensorMeta

    func = ForwardTraceOnly()
    shape1, shape2 = ([32, 1, 4, 8], [1, 8, 12, 16])
    meta1, meta2 = (TensorMeta(5, 1, shape1), TensorMeta(3, 8, shape2))
    input_tensor1 = TracedTensor.from_torch_tensor(torch.Tensor(size=shape1), meta1)
    input_tensor2 = TracedTensor.from_torch_tensor(torch.Tensor(size=shape2), meta2)

    # 1 -> 1
    output_tensor = func(torch.Tensor.view, input_tensor1, [-1])
    assert output_tensor.tensor_meta != input_tensor1.tensor_meta
    assert output_tensor.tensor_meta.shape == (1024, )

    # 1 -> N
    outputs = func(torch.Tensor.chunk, input_tensor1, 3)
    for out in outputs:
        assert out.tensor_meta == input_tensor1.tensor_meta

    # N -> N (2 -> 2)
    outputs = func(lambda x: x + [5], [input_tensor1, input_tensor2])
    assert outputs[0].tensor_meta == input_tensor1.tensor_meta
    assert outputs[1].tensor_meta == input_tensor2.tensor_meta

    # M -> N (2 -> 3)
    with pytest.raises(RuntimeError):
        outputs = func(lambda x: x + [torch.Tensor(shape2)], [input_tensor1, input_tensor2])

    # M -> N (2 -> 1)
    with pytest.raises(RuntimeError):
        outputs = func(lambda x: x[0], [input_tensor1, input_tensor2])


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
        args = (mock_tensor, )
        kwargs = {}
        args, kwargs = ModelForTest.simple_wrap_fn(args, kwargs)
        return model(*args, **kwargs)


input_shapes = [
    (1, 3, 224, 224),
    (2, 3, 224, 224),
    (1, 3, 500, 500)
]


@pytest.mark.parametrize("input_shape", input_shapes)
def test_activation_shape_tracing(input_shape: Tuple):
    model = ModelForTest()
    input_info = ModelInputInfo(input_shape)
    graph_builder = GraphBuilder(create_dummy_forward_fn([input_info, ], with_input_tracing=True,
                                                         with_output_tracing=True))
    graph = graph_builder.build_graph(model)

    shape1 = (input_shape[0], ModelForTest.CONV1_OUT_CHANNELS, input_shape[2], input_shape[3])
    final_shape = (input_shape[0], ModelForTest.OUT_CHANNELS, input_shape[2], input_shape[3])
    ref_maxpool_out_edge_shapes = [(shape1[0], shape1[1],
                                        shape1[2] // ModelForTest.MAXPOOL_SIZE,
                                        shape1[3] // ModelForTest.MAXPOOL_SIZE)]
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
        input_edges = graph.get_nncf_graph_pattern_io([node_id, ]).input_edges
        output_edges = graph.get_nncf_graph_pattern_io([node_id, ]).output_edges
        input_tensor_shapes = [x.tensor_shape for x in input_edges]
        output_tensor_shapes = [x.tensor_shape for x in output_edges]
        assert input_tensor_shapes == ref_input_shapes, "Failed for node ID: {}".format(node_id)
        assert output_tensor_shapes == ref_output_shapes, "Failed for node ID: {}".format(node_id)


TEST_KEYWORD_1 = "keyword1"
TEST_KEYWORD_2 = "keyword2"
INPUT_INFO_CONFIG_VS_FORWARD_ARGS = [
    ({"sample_size": [2, 3, 300, 300],
      "type": "float",
      "filler": "zeros"},
     [ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS)]),

    ([{"sample_size": [1, 128],
       "type": "long",
       "filler": "ones"},
      {"sample_size": [1, 128],
       "type": "long",
       "filler": "ones"},
      {"sample_size": [1, 128],
       "type": "long",
       "filler": "zeros"}], [ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES),
                             ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES),
                             ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES), ]),

    ([{"sample_size": [2, 3, 300, 300],
       "type": "float",
       "filler": "zeros"},
      {"sample_size": [1, 128],
       "type": "long",
       "filler": "ones",
       "keyword": TEST_KEYWORD_1}],
     [ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS),
      ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_1)]),

    ([{"sample_size": [8, 7],
       "type": "float",
       "filler": "random",
       "keyword": TEST_KEYWORD_1},
      {"sample_size": [2, 3, 300, 300],
       "type": "float",
       "filler": "zeros"},
      {"sample_size": [1, 128],
       "type": "long",
       "filler": "ones",
       "keyword": TEST_KEYWORD_2}, ],
     [ModelInputInfo([2, 3, 300, 300], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ZEROS),
      ModelInputInfo([8, 7], type_str="float", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_1),
      ModelInputInfo([1, 128], type_str="long", filler=ModelInputInfo.FILLER_TYPE_ONES, keyword=TEST_KEYWORD_2)]),
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
    compressed_model, compression_ctrl = \
        create_compressed_model_and_algo_for_test(model, config,
                                                  dummy_forward_fn=ModelForTest.simple_user_dummy_forward,
                                                  wrap_inputs_fn=ModelForTest.simple_wrap_fn)
    return compressed_model, compression_ctrl


def create_model_with_user_wrap_inputs_fn():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, compression_ctrl = \
        create_compressed_model_and_algo_for_test(model, config,
                                                  dummy_forward_fn=ModelForTest.simple_user_dummy_forward,
                                                  wrap_inputs_fn=ModelForTest.simple_wrap_fn)
    return compressed_model, compression_ctrl


class TestGraphStability:
    MODEL_CREATORS_AND_IDS = [
        (create_model_and_control_with_defaults, 'default'),
        (create_model_with_user_dummy, 'user_dummy'),
        (create_model_with_user_wrap_inputs_fn, 'user_wrap_inputs_fn')
    ]

    @pytest.fixture(params=[x[0] for x in MODEL_CREATORS_AND_IDS],
                    ids=[x[1] for x in MODEL_CREATORS_AND_IDS])
    def model_and_ctrl_creator(self, request):
        return request.param

    def test_dynamic_graph_does_not_inflate_during_multiple_forwards(self, model_and_ctrl_creator):
        compressed_model, _ = model_and_ctrl_creator()
        input_tensor = torch.zeros(input_shapes[0])
        ref_graph = deepcopy(compressed_model.get_dynamic_graph())
        for _ in range(0, 10):
            _ = compressed_model(input_tensor)
            curr_graph = compressed_model.get_dynamic_graph()
            assert curr_graph == ref_graph

    def test_dynamic_graph_is_the_same_after_export(self, model_and_ctrl_creator, tmp_path):
        compressed_model, ctrl = model_and_ctrl_creator()
        ref_graph = deepcopy(compressed_model.get_dynamic_graph())
        ctrl.export_model('tmp.onnx')
        curr_graph = compressed_model.get_dynamic_graph()
        assert curr_graph == ref_graph

    def test_dummy_forwards_do_not_inflate_dynamic_graph(self, model_and_ctrl_creator):
        compressed_model, _ = model_and_ctrl_creator()
        ref_graph = deepcopy(compressed_model.get_dynamic_graph())
        compressed_model.do_dummy_forward()
        curr_graph = deepcopy(compressed_model.get_dynamic_graph())
        assert curr_graph == ref_graph

    def test_compressed_graph_with_user_wrap_fn(self):
        # Create a model with a dummy forward analogous to
        # the default dummy forward, compare original and compressed model graphs afterwards

        comp_model_wo_wrap, _ = create_model_and_control_with_defaults()
        comp_model_w_wrap, _ = create_model_with_user_wrap_inputs_fn()

        ref_original_graph = comp_model_wo_wrap.get_graph()
        ref_compressed_graph = comp_model_wo_wrap.get_graph()

        original_graph_with_wrap = comp_model_w_wrap.get_graph()
        compressed_graph_with_wrap = comp_model_w_wrap.get_graph()

        assert ref_original_graph == original_graph_with_wrap
        assert ref_compressed_graph == compressed_graph_with_wrap

    def test_compressed_graph_with_user_dummy_forward(self):
        # Create a model with a dummy forward analogous to
        # the default dummy forward, compare original and compressed model graphs afterwards

        comp_model_wo_dummy, _ = create_model_and_control_with_defaults()
        comp_model_w_dummy, _ = create_model_with_user_dummy()

        ref_original_graph = comp_model_wo_dummy.get_graph()
        ref_compressed_graph = comp_model_wo_dummy.get_graph()

        original_graph_with_dummy = comp_model_w_dummy.get_graph()
        compressed_graph_with_dummy = comp_model_w_dummy.get_graph()

        assert ref_original_graph == original_graph_with_dummy
        assert ref_compressed_graph == compressed_graph_with_dummy


def test_nncf_graph_auxiliary_node_structure():
    model = ModelForTest()
    config = get_basic_quantization_config("symmetric", input_sample_sizes=list(input_shapes[0]))
    register_bn_adaptation_init_args(config)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    nncf_graph = compressed_model.get_graph()

    input_nodes = nncf_graph.get_input_nodes()
    output_nodes = nncf_graph.get_output_nodes()

    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

    assert input_nodes[0].node_type == NNCFGraphNodeType.INPUT_NODE
    assert output_nodes[0].node_type == NNCFGraphNodeType.OUTPUT_NODE


def test_get_all_nodes():
    model = ModelForNameTest()
    ref_list = [
        'ModelForNameTest/Conv2d[conv1]/conv2d_0',
        'ModelForNameTest/BatchNorm2d[bn1]/batch_norm_0',
        'ModelForNameTest/ReLU/relu_0',
        'ModelForNameTest/relu_0',
        'ModelForNameTest/BatchNorm2d[bn2]/batch_norm_0',
        'ModelForNameTest/Sequential[layer2]/Sequential[layer1]/Conv2d[conv01]/conv2d_0',
        'ModelForNameTest/Sequential[layer2]/Sequential[layer1]/BatchNorm2d[norm01]/batch_norm_0',
        'ModelForNameTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]/relu_0',
        'ModelForNameTest/Sequential[layer2]/Sequential[layer1]/MaxPool2d[pool01]/max_pool2d_0',
        'ModelForNameTest/Sequential[layer2]/Conv2d[conv02]/conv2d_0',
        'ModelForNameTest/Sequential[layer2]/ReLU[relu02]/relu_0',
        'ModelForNameTest/Sequential[layer2]/BatchNorm2d[norm02]/batch_norm_0',
        'ModelForNameTest/Sequential[layer2]/MaxPool2d[pool02]/max_pool2d_0',
        'ModelForNameTest/AvgPool2d[avgpool]/avg_pool2d_0'
    ]

    builder = GraphBuilder(create_dummy_forward_fn([ModelInputInfo((1, 1, 4, 4)), ]))
    graph = builder.build_graph(model)
    test_list = [node_name.split(' ', 1)[1] for node_name in graph.get_all_node_keys()]
    assert ref_list == test_list
