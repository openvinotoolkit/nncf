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

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import opset9 as opset

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_inplace_batch_mean_op
from nncf.openvino.graph.node_utils import get_inplace_max_op
from nncf.openvino.graph.node_utils import get_inplace_mean_op
from nncf.openvino.graph.node_utils import get_inplace_mean_per_ch
from nncf.openvino.graph.node_utils import get_inplace_min_op
from nncf.openvino.graph.node_utils import get_ov_model_reduce_node_name
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVInplaceFnInsertionCommand
from nncf.openvino.graph.transformations.commands import OVNullBiasInsertionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import ConvNotBiasModel
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import QuantizedModel
from tests.openvino.native.models import SimpleSplitModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.models import ZeroRankEltwiseModel

REFERENCE_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / "data" / "reference_graphs" / "original_nncf_graph"

TARGET_INSERT_LAYERS = [["Add"], ["MatMul"], ["Add", "MatMul"]]
TARGET_PRE_LAYER_FQS = [["Add/fq_input_0"], ["MatMul/fq_input_0"], ["Add/fq_input_0", "MatMul/fq_input_0"]]
TARGET_POST_LAYER_FQS = [["Add/fq_output_0"], ["MatMul/fq_output_0"], ["Add/fq_output_0", "MatMul/fq_output_0"]]
TARGET_WEIGHTS_FQS = [["Add/fq_weights_1"], ["MatMul/fq_weights_1"], ["Add/fq_weights_1", "MatMul/fq_weights_1"]]


def create_transformed_model(model, target_layers, target_type, command_type, port_id=0, **kwargs):
    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = OVTargetPoint(target_type, target_layer, port_id=port_id)
        command = command_type(target_point, **kwargs)
        transformation_layout.register(command)

    model_transformer = OVModelTransformer(model)
    transformed_model = model_transformer.transform(transformation_layout)
    return transformed_model


def get_extra_outputs(original_model, transformed_model):
    extra_outputs = set()
    for out in transformed_model.get_results():
        extra_outputs.add(out.get_friendly_name())

    for out in original_model.get_results():
        extra_outputs.remove(out.get_friendly_name())

    return extra_outputs


def get_fq_nodes(model):
    fq_nodes = []
    for op in model.get_ops():
        if op.get_type_name() == "FakeQuantize":
            fq_nodes.append(op.get_friendly_name())

    return fq_nodes


@dataclass
class InplaceOpTestCase:
    name: str
    reduce_shape: Tuple[int, ...]
    op_builder: Callable
    ref_types: List[str]
    ref_values: List[np.array]
    dims: str = "DEFAULT"

    def __str__(self) -> str:
        return str(self.__dict__.values())


LINEAR_MODEL_SHAPES = {
    "DEFAULT": {
        "input_shape": [1, 3, 2, 8],
        "reshape_shape": [1, 3, 4, 4],
        "matmul_w_shape": [1, 3, 4, 4],
        "add_shape": [1, 3, 4, 4],
    },
    "SHORT": {"input_shape": [1, 3, 2, 8], "reshape_shape": [48], "matmul_w_shape": [48, 48], "add_shape": [48]},
}
INPLACE_OPS_TEST_CASES = [
    # Forwarded reduce shape
    InplaceOpTestCase("min", (1, 2), get_inplace_min_op, ["ReduceMin"], [(1, 2)]),
    InplaceOpTestCase("mean", (1, 2), get_inplace_mean_op, ["ReduceMean"], [(1, 2)]),
    InplaceOpTestCase("max", (1, 2), lambda o, r: get_inplace_max_op(o, r, False), ["ReduceMax"], [(1, 2)]),
    InplaceOpTestCase(
        "abs_max", (1, 2), lambda o, r: get_inplace_max_op(o, r, True), ["Abs", "ReduceMax"], [None, (1, 2)]
    ),
    # Calculated reduce shape
    InplaceOpTestCase("min", None, get_inplace_min_op, ["ReduceMin"], [(0, 1, 2, 3)]),
    InplaceOpTestCase("mean", None, get_inplace_mean_op, ["ReduceMean"], [(0, 1, 2, 3)]),
    InplaceOpTestCase("max", None, lambda o, r: get_inplace_max_op(o, r, False), ["ReduceMax"], [(0, 1, 2, 3)]),
    InplaceOpTestCase(
        "abs_max", None, lambda o, r: get_inplace_max_op(o, r, True), ["Abs", "ReduceMax"], [None, (0, 1, 2, 3)]
    ),
    # Batch mean and mean per ch operations
    InplaceOpTestCase("batch_mean", None, lambda o, r: get_inplace_batch_mean_op(o), ["ReduceMean"], [(0,)]),
    InplaceOpTestCase("mean_per_ch", 1, get_inplace_mean_per_ch, ["Reshape", "ReduceMean"], [(1, 3, 16), (0, 2)]),
    InplaceOpTestCase(
        "mean_per_ch",
        2,
        get_inplace_mean_per_ch,
        ["Transpose", "Reshape", "ReduceMean"],
        [(0, 2, 1, 3), (1, 4, 12), (0, 2)],
    ),
    InplaceOpTestCase("mean_per_ch", 0, get_inplace_mean_per_ch, ["ReduceMean"], [(0,)], dims="SHORT"),
]


def get_prev_node(node, input_port):
    return node.input(input_port).get_source_output().get_node()


def get_next_nodes(node, output_port):
    return [x.get_node() for x in node.output(output_port).target_inputs]


def get_node_by_name(model: ov.Model, name: str):
    nodes = [node for node in model.get_ops() if node.get_friendly_name() == name]
    assert len(nodes) == 1
    return nodes[0]


def check_inplace_op(target_node, ref_types, ref_vals, inplace_branches_num, output_port_id):
    next_nodes = get_next_nodes(target_node, output_port_id)
    first_inplace_op = [node for node in next_nodes if node.type_info.name == ref_types[0]]
    assert len(first_inplace_op) == inplace_branches_num
    node = first_inplace_op[0]
    for t, ref_val in zip(ref_types, ref_vals):
        assert node.type_info.name == t
        if ref_val is not None:
            const = get_prev_node(node, 1)
            if ref_val == []:
                assert const.get_data().shape == (0,)
            else:
                res = np.equal(const.get_data(), np.array(ref_val))
                assert all(res)

        nodes = get_next_nodes(node, 0)
        assert len(nodes) == 1
        node = nodes[0]


@pytest.mark.parametrize("test_params", INPLACE_OPS_TEST_CASES, ids=[str(case) for case in INPLACE_OPS_TEST_CASES])
@pytest.mark.parametrize(
    "target_type", [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]
)
@pytest.mark.parametrize("target_layers", TARGET_INSERT_LAYERS)
def test_inplace_fn_insertion(test_params: InplaceOpTestCase, target_type, target_layers):
    dims = LINEAR_MODEL_SHAPES[test_params.dims]
    model = LinearModel(**dims).ov_model
    port_id = 1 if target_type == TargetType.OPERATION_WITH_WEIGHTS else 0
    transformed_model = create_transformed_model(
        model,
        target_layers,
        target_type,
        OVInplaceFnInsertionCommand,
        port_id=port_id,
        inplace_op_fn=test_params.op_builder(test_params.name, test_params.reduce_shape),
        fn_output_port_id=0,
    )

    inplace_branches_num = 1
    if target_type == TargetType.PRE_LAYER_OPERATION:
        inplace_branches_num = len(target_layers)
        target_layers = ["Reshape"]

    target_nodes = []
    for target_layer in target_layers:
        target_node = get_node_by_name(transformed_model, target_layer)
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            source_output = target_node.input(1).get_source_output()
            target_node = source_output.get_node()
            port_id = source_output.get_index()
        check_inplace_op(target_node, test_params.ref_types, test_params.ref_values, inplace_branches_num, port_id)
        target_nodes.append((target_node, port_id))

    default_output_fn_port = 0
    extra_outputs = get_extra_outputs(model, transformed_model)
    ref_output_names = [
        get_result_node_name(
            get_ov_model_reduce_node_name(target_node.get_friendly_name(), test_params.name, port_id),
            default_output_fn_port,
        )
        for target_node, port_id in target_nodes
    ]
    assert len(extra_outputs) == len(ref_output_names)
    for out_name in extra_outputs:
        assert out_name in ref_output_names


SPLIT_MODEL_INPUT_SHAPES = {
    "DEFAULT": {"input_shape": [1, 9, 4, 4], "splits": 3},
    "SHORT": {"input_shape": [1, 9], "splits": 3},
}


@pytest.mark.parametrize("test_params", INPLACE_OPS_TEST_CASES, ids=[str(case) for case in INPLACE_OPS_TEST_CASES])
def test_split_inplace_fn_insertion(test_params: InplaceOpTestCase):
    dims = SPLIT_MODEL_INPUT_SHAPES[test_params.dims]
    model = SimpleSplitModel(**dims).ov_model
    target_layer = "Split"
    port_id = 1
    transformed_model = create_transformed_model(
        model,
        [target_layer],
        TargetType.POST_LAYER_OPERATION,
        OVInplaceFnInsertionCommand,
        port_id=port_id,
        inplace_op_fn=test_params.op_builder(test_params.name, test_params.reduce_shape),
        fn_output_port_id=0,
    )

    target_node = get_node_by_name(transformed_model, target_layer)
    check_inplace_op(target_node, test_params.ref_types, test_params.ref_values, 1, port_id)

    default_output_fn_port = 0
    extra_outputs = get_extra_outputs(model, transformed_model)
    ref_output_name = get_result_node_name(
        get_ov_model_reduce_node_name(target_node.get_friendly_name(), test_params.name, port_id),
        default_output_fn_port,
    )
    assert len(extra_outputs) == 1
    assert ref_output_name in extra_outputs


@pytest.mark.parametrize(
    "input_shape,raise_error",
    [((ov.Dimension(), 3, 3, 3), False), ((1, 3, 3, ov.Dimension()), False), (ov.PartialShape("?").dynamic(), True)],
)
def test_inplace_reduce_fn_dynamic_shapes(input_shape, raise_error):
    input_1 = opset.parameter(input_shape, name="Input")
    fn = get_inplace_min_op("test", reduction_shape=None)
    if raise_error:
        with pytest.raises(RuntimeError):
            fn(input_1, 0)
        return
    op = fn(input_1, 0)
    # check_const
    ref_const = np.array([0, 1, 2, 3])
    assert all(np.equal(get_prev_node(op, 1).get_data(), ref_const))


@pytest.mark.parametrize("reduction_shape", [None, np.array([], dtype=np.int64)])
def test_inplace_reduce_fn_zero_rank_output(reduction_shape):
    model = ZeroRankEltwiseModel().ov_model
    target_layer = "Add"
    port_id = 1
    name = "min"
    transformed_model = create_transformed_model(
        model,
        [target_layer],
        TargetType.OPERATION_WITH_WEIGHTS,
        OVInplaceFnInsertionCommand,
        port_id=port_id,
        inplace_op_fn=get_inplace_min_op(name, reduction_shape=reduction_shape),
        fn_output_port_id=0,
    )
    target_node = get_prev_node(get_node_by_name(transformed_model, target_layer), 1)
    check_inplace_op(target_node, ["ReduceMin"], [[]], 1, 0)
    extra_outputs = get_extra_outputs(model, transformed_model)
    ref_output_name = get_result_node_name(get_ov_model_reduce_node_name(target_node.get_friendly_name(), name, 0), 0)
    assert len(extra_outputs) == 1
    assert extra_outputs.pop() == ref_output_name


DYNAMIC_SHAPE_TEST_CASES = [
    InplaceOpTestCase("mean_per_ch", 1, get_inplace_mean_per_ch, ["Reshape"], [(-1, 3, 9), (0, 2)]),
    InplaceOpTestCase("mean_per_ch", 1, get_inplace_mean_per_ch, ["Reshape"], [(1, 3, -1), (0, 2)]),
    InplaceOpTestCase("mean_per_ch", 3, get_inplace_mean_per_ch, ["Transpose", "Reshape"], [(0, 3, 2, 1), (-1, 3, 9)]),
    InplaceOpTestCase("mean_per_ch", 3, get_inplace_mean_per_ch, ["Transpose", "Reshape"], [(0, 3, 2, 1), (1, -1, 9)]),
    InplaceOpTestCase("mean_per_ch", 3, get_inplace_mean_per_ch, ["Transpose", "Reshape"], [(0, 3, 2, 1), (1, 3, -1)]),
]


@pytest.mark.parametrize(
    "test_params,input_shape,raise_error",
    [
        (DYNAMIC_SHAPE_TEST_CASES[0], (ov.Dimension(), 3, 3, 3), False),
        (DYNAMIC_SHAPE_TEST_CASES[1], (1, 3, 3, ov.Dimension()), False),
        (DYNAMIC_SHAPE_TEST_CASES[0], (ov.Dimension(), ov.Dimension(), 3, 3), True),
        (DYNAMIC_SHAPE_TEST_CASES[1], (1, 3, ov.Dimension(), ov.Dimension()), False),
        (DYNAMIC_SHAPE_TEST_CASES[1], (1, ov.Dimension(), ov.Dimension(), 3), True),
        (DYNAMIC_SHAPE_TEST_CASES[2], (ov.Dimension(), 3, 3, 3), False),
        (DYNAMIC_SHAPE_TEST_CASES[3], (1, 3, 3, ov.Dimension()), False),
        (DYNAMIC_SHAPE_TEST_CASES[4], (1, ov.Dimension(), ov.Dimension(), 3), False),
        (DYNAMIC_SHAPE_TEST_CASES[4], (1, ov.Dimension(), ov.Dimension(), ov.Dimension()), True),
    ],
)
def test_inplace_mean_per_ch_fn_dynamic_shapes(test_params: InplaceOpTestCase, input_shape, raise_error):
    input_1 = opset.parameter(input_shape, name="Input")
    fn = test_params.op_builder(test_params.name, test_params.reduce_shape)
    if raise_error:
        with pytest.raises(RuntimeError):
            fn(input_1, 0)
        return
    fn(input_1, 0)
    check_inplace_op(input_1, test_params.ref_types, test_params.ref_values, 1, 0)


@pytest.mark.parametrize(
    "target_type", [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]
)
@pytest.mark.parametrize("target_layers", TARGET_INSERT_LAYERS)
def test_output_insertion(target_type, target_layers):
    model = LinearModel().ov_model
    port_id = 1 if target_type == TargetType.OPERATION_WITH_WEIGHTS else 0
    transformed_model = create_transformed_model(
        model, target_layers, target_type, OVOutputInsertionCommand, port_id=port_id
    )

    if target_type == TargetType.PRE_LAYER_OPERATION:
        target_layers = ["Reshape"]

    target_nodes = []
    for target_layer in target_layers:
        target_node = get_node_by_name(transformed_model, target_layer)
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            source_output = target_node.input(1).get_source_output()
            target_node = source_output.get_node()
            port_id = source_output.get_index()
        target_nodes.append((target_node, port_id))

    extra_outputs = get_extra_outputs(model, transformed_model)
    ref_output_names = [
        get_result_node_name(target_node.get_friendly_name(), port_id) for target_node, port_id in target_nodes
    ]
    assert len(extra_outputs) == len(ref_output_names)
    for out_name in extra_outputs:
        assert out_name in ref_output_names


@pytest.mark.parametrize("test_params", INPLACE_OPS_TEST_CASES, ids=[str(case) for case in INPLACE_OPS_TEST_CASES])
def test_split_output_insertion(test_params: InplaceOpTestCase):
    dims = SPLIT_MODEL_INPUT_SHAPES[test_params.dims]
    model = SimpleSplitModel(**dims).ov_model
    target_layer = "Split"
    port_id = 1
    transformed_model = create_transformed_model(
        model, [target_layer], TargetType.POST_LAYER_OPERATION, OVOutputInsertionCommand, port_id=port_id
    )

    target_node = get_node_by_name(transformed_model, target_layer)
    extra_outputs = get_extra_outputs(model, transformed_model)
    ref_output_name = get_result_node_name(target_node.get_friendly_name(), port_id)
    assert len(extra_outputs) == 1
    assert ref_output_name in extra_outputs


TARGET_LAYERS = [("Conv_1/fq_input_0", "Concat_1/fq_input_0", "Conv_3/fq_weights_0", "Add_2/fq_weights_0")]


@pytest.mark.parametrize("target_layers", TARGET_LAYERS)
def test_node_removing(target_layers):
    model_to_test = QuantizedModel()
    model = model_to_test.ov_model

    transformation_layout = TransformationLayout()

    for target_layer in target_layers:
        target_point = OVTargetPoint(TargetType.LAYER, target_layer, 0)
        command = OVFQNodeRemovingCommand(target_point)
        transformation_layout.register(command)

    model_transformer = OVModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    ref_name = "removed_nodes_in_" + model_to_test.ref_graph_name
    compare_nncf_graphs(transformed_model, REFERENCE_GRAPHS_DIR / ref_name)


@pytest.mark.parametrize("target_layers, ref_fq_names", zip(TARGET_INSERT_LAYERS, TARGET_PRE_LAYER_FQS))
def test_fq_insertion_pre_layer(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = FakeQuantizeParameters(min_values, max_values, min_values, max_values, levels=256)

    transformed_model = create_transformed_model(
        model,
        target_layers,
        TargetType.PRE_LAYER_OPERATION,
        OVQuantizerInsertionCommand,
        quantizer_parameters=quantizer_parameters,
    )
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


@pytest.mark.parametrize("target_layers, ref_fq_names", zip(TARGET_INSERT_LAYERS, TARGET_POST_LAYER_FQS))
def test_fq_insertion_post_layer(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = FakeQuantizeParameters(min_values, max_values, min_values, max_values, levels=256)
    transformed_model = create_transformed_model(
        model,
        target_layers,
        TargetType.POST_LAYER_OPERATION,
        OVQuantizerInsertionCommand,
        quantizer_parameters=quantizer_parameters,
    )
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


@pytest.mark.parametrize("target_layers, ref_fq_names", zip(TARGET_INSERT_LAYERS, TARGET_WEIGHTS_FQS))
def test_fq_insertion_weights(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = FakeQuantizeParameters(min_values, max_values, min_values, max_values, levels=256)
    transformed_model = create_transformed_model(
        model,
        target_layers,
        TargetType.OPERATION_WITH_WEIGHTS,
        OVQuantizerInsertionCommand,
        port_id=1,
        quantizer_parameters=quantizer_parameters,
    )
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


MODELS_WITH_PARAMETERS = [
    {
        "model": ConvModel().ov_model,
        "layers": ["Conv"],
        "values": [np.full((3,), 2)],
        "refs": [2.0],
    },
    {
        "model": FPModel(const_dtype="FP16").ov_model,
        "layers": ["MatMul"],
        "values": [np.full((3,), 2)],
        "refs": [2.0],
    },
]


@pytest.mark.parametrize("model_with_parameters", MODELS_WITH_PARAMETERS)
def test_bias_correction(model_with_parameters):
    model = model_with_parameters["model"]
    layers = model_with_parameters["layers"]
    values = model_with_parameters["values"]
    refs = model_with_parameters["refs"]

    transformed_model = create_transformed_model(
        model, layers, TargetType.LAYER, OVBiasCorrectionCommand, port_id=1, **{"bias_value": values}
    )
    ops_dict = {op.get_friendly_name(): op for op in transformed_model.get_ops()}

    for node_name, bias_reference in zip(layers, refs):
        node = ops_dict[node_name]
        node_inputs = [port.get_node() for port in node.output(0).get_target_inputs()]
        node_with_bias = node_inputs[0]

        potential_bias = node_with_bias.input_value(1).node
        if potential_bias.get_type_name() == "Convert":
            potential_bias = potential_bias.input_value(0).node
        assert potential_bias.get_type_name() == "Constant"
        assert np.all(potential_bias.get_vector() == bias_reference)


def test_no_transformations():
    def infer_model_with_ones(model, shape):
        ie = ov.Core()
        compiled_model = ie.compile_model(model)
        _input = np.ones(shape)
        model_outputs = compiled_model(_input)
        return {out.get_node().get_friendly_name(): data for out, data in model_outputs.items()}

    model = LinearModel().ov_model
    input_shape = [1, 3, 4, 2]
    model_transformer = OVModelTransformer(model)
    transformed_model = model_transformer.transform(TransformationLayout())

    ret_val_1 = infer_model_with_ones(model, input_shape)
    ret_val_2 = infer_model_with_ones(transformed_model, input_shape)
    assert ret_val_1.keys() == ret_val_2.keys()
    for output in ret_val_1.keys():
        assert np.allclose(ret_val_1[output], ret_val_2[output])
    assert id(transformed_model) != id(model)


MODELS_WITH_PARAMETERS = [
    {"model": ConvNotBiasModel().ov_model, "layers": ["Conv"]},
    {"model": WeightsModel().ov_model, "layers": ["Conv", "Conv_backprop"]},
]


@pytest.mark.parametrize("model_with_parameters", MODELS_WITH_PARAMETERS)
def test_null_biases_insertion(model_with_parameters):
    model = model_with_parameters["model"]
    layers = model_with_parameters["layers"]

    transformed_model = create_transformed_model(model, layers, TargetType.LAYER, OVNullBiasInsertionCommand, port_id=0)
    ops_dict = {op.get_friendly_name(): op for op in transformed_model.get_ops()}

    for layer_name in layers:
        node = ops_dict[layer_name]
        layer_shape = ops_dict[layer_name].shape
        bias_dtype = node.get_element_type().to_dtype()

        # We assume that there is only ONE bias after convolution
        output_port = node.output(0)
        add_with_bias = list(output_port.get_target_inputs())[0].get_node()
        assert add_with_bias.get_type_name() == "Add"

        # We assume that the bias inserts only on 1st position for Add layer
        bias_node = add_with_bias.input(1).get_source_output().get_node()
        assert bias_node.get_type_name() == "Constant"

        assert all(bias_node.get_vector() == np.zeros(layer_shape[1], dtype=bias_dtype))
