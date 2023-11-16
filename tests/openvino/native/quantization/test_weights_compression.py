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

import os
from typing import Callable, List

import numpy as np
import openvino.runtime as ov
import pytest
from attr import dataclass

from nncf import CompressWeightsMode
from nncf.openvino.graph.node_utils import get_const_value
from nncf.quantization import compress_weights
from nncf.quantization.algorithms.weight_compression.openvino_backend import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.openvino_backend import _get_integer_quantization_error
from nncf.quantization.algorithms.weight_compression.openvino_backend import _reshape_weights_for_grouped_quantization
from nncf.scopes import IgnoredScope
from tests.openvino.native.common import get_openvino_version
from tests.openvino.native.models import IntegerModel
from tests.openvino.native.models import SequentialMatmulModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.quantization.test_fq_params_calculation import REFERENCE_SCALES_DIR
from tests.shared.helpers import compare_stats
from tests.shared.helpers import dump_to_json
from tests.shared.helpers import load_json

TEST_MODELS = {
    IntegerModel: ["matmul_2_data", "gather_2_data", "matmul_1_data"],
    WeightsModel: ["weights_0", "weights_1"],
}


def get_next_node(node):
    target_inputs = node.output(0).get_target_inputs()
    assert len(target_inputs) == 1
    next_node = next(iter(target_inputs)).get_node()
    return next_node


def check_int8_node(op: ov.Node):
    assert op.get_element_type() == ov.Type(np.uint8)
    compressed_weight = get_const_value(op)

    convert_node = get_next_node(op)
    assert convert_node.get_type_name() == "Convert"

    sub_node = get_next_node(convert_node)
    assert sub_node.get_type_name() == "Subtract"

    convert_node = sub_node.input_value(1).get_node()
    assert convert_node.get_type_name() == "Convert"

    zero_point_node = convert_node.input_value(0).get_node()
    zero_point = get_const_value(zero_point_node)

    mul_node = get_next_node(sub_node)
    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    scale = get_const_value(scale_node)

    return {
        "compressed_weight": compressed_weight,
        "zero_point": zero_point,
        "scale": scale,
    }


def check_int4_grouped(op: ov.Node, mode: CompressWeightsMode, group_size: int = 3):
    assert op.get_element_type() == ov.Type.u4
    weight_shape = op.shape
    # NOTE: get_const_value doesn't work for 4-bit types
    assert list(weight_shape)[-1] == group_size
    reduced_weight_shape = list(weight_shape)
    reduced_weight_shape[-1] = 1

    convert_node = get_next_node(op)
    assert convert_node.get_type_name() == "Convert"

    sub_node = get_next_node(convert_node)
    assert sub_node.get_type_name() == "Subtract"

    convert_node = sub_node.input_value(1).get_node()
    assert convert_node.get_type_name() == "Convert"

    zero_point_node = convert_node.input_value(0).get_node()
    assert zero_point_node.get_element_type() == ov.Type.u4
    if mode == CompressWeightsMode.INT4_SYM:
        assert list(zero_point_node.shape) == [1]
    else:
        assert list(zero_point_node.shape) == reduced_weight_shape

    mul_node = get_next_node(sub_node)
    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    assert list(scale_node.shape) == reduced_weight_shape

    reshape_node = get_next_node(mul_node)
    assert reshape_node.get_type_name() == "Reshape"

    return {
        "scale": get_const_value(scale_node),
    }


def check_nf4_grouped(op: ov.Node, group_size: int = 3):
    assert op.get_element_type() == ov.Type.nf4
    weight_shape = op.shape
    # NOTE: get_const_value doesn't work for 4-bit types
    assert list(weight_shape)[-1] == group_size
    reduced_weight_shape = list(weight_shape)
    reduced_weight_shape[-1] = 1

    convert_node = get_next_node(op)
    assert convert_node.get_type_name() == "Convert"

    mul_node = get_next_node(convert_node)
    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    assert list(scale_node.shape) == reduced_weight_shape

    reshape_node = get_next_node(mul_node)
    assert reshape_node.get_type_name() == "Reshape"

    return {
        "scale": get_const_value(scale_node),
    }


def check_int4_sym_grouped(op: ov.Node):
    return check_int4_grouped(op, mode=CompressWeightsMode.INT4_SYM)


def check_int4_asym_grouped(op: ov.Node):
    return check_int4_grouped(op, mode=CompressWeightsMode.INT4_ASYM)


def get_mixed_mapping(primary_fn: Callable, list_layers: List[str]):
    mapping = {node_name: check_int8_node for node_name in list_layers}

    for node_name in TEST_MODELS[IntegerModel][1:-1]:
        mapping[node_name] = primary_fn
    return mapping


@pytest.mark.parametrize(
    ("mode", "group_size", "check_fn_per_node_map"),
    (
        (CompressWeightsMode.INT8, -1, {node_name: check_int8_node for node_name in TEST_MODELS[IntegerModel]}),
        (CompressWeightsMode.INT4_SYM, 3, get_mixed_mapping(check_int4_sym_grouped, TEST_MODELS[IntegerModel])),
        (CompressWeightsMode.INT4_ASYM, 3, get_mixed_mapping(check_int4_asym_grouped, TEST_MODELS[IntegerModel])),
        (CompressWeightsMode.NF4, 3, get_mixed_mapping(check_nf4_grouped, TEST_MODELS[IntegerModel])),
    ),
)
def test_compare_compressed_weights(mode, group_size, check_fn_per_node_map):
    ov_version = get_openvino_version()
    if mode == CompressWeightsMode.NF4 and ov_version != "2023.2":
        pytest.xfail("NF4 is not supported until 2023.2")
    model = IntegerModel().ov_model
    compressed_model = compress_weights(model, mode=mode, group_size=group_size)
    actual_stats = {}
    for op in compressed_model.get_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and op_name in check_fn_per_node_map:
            check_fn = check_fn_per_node_map[op_name]
            actual_stats[op_name] = check_fn(op)

    ref_stats_path = REFERENCE_SCALES_DIR / f"IntegerModel_compressed_weights_{mode.value}.json"

    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        dump_to_json(ref_stats_path, actual_stats)

    ref_stats = load_json(ref_stats_path)
    compare_stats(ref_stats, actual_stats)


@pytest.mark.parametrize("group_size", (1, 3))
@pytest.mark.parametrize(
    ("ratio", "ref_nf4_nodes"),
    (
        (1, ["weights_1", "weights_2", "weights_3"]),
        (0.8, ["weights_2", "weights_3"]),
        (0.4, ["weights_3"]),
        (0.3, []),
    ),
)
def test_mixed_precision(ratio, group_size, ref_nf4_nodes):
    if ratio > 0.3:
        pytest.xfail("Waiting for the merge NF4 support in OV - PR 19900")
    model = SequentialMatmulModel().ov_model
    compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4, ratio=ratio, group_size=group_size)
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_nf4_nodes:
            assert op.get_element_type() == ov.Type.nf4


@dataclass
class QuantErrorDesc:
    weight: List[float]
    ref_error: int = 0
    axis = (1,)
    name: str = ""
    atol: float = None
    config: WeightCompressionConfig = WeightCompressionConfig()

    def __str__(self):
        prefix = "exact_match_" if self.ref_error == 0 else ""
        name = self.name.replace(" ", "_") if self.name else self.__class__.__name__
        return prefix + name


SCALE_1 = 1.2
SCALE_2 = 3.4
SCALE_3 = 5.6
SCALE_4 = 7.8
LINSPACE = np.arange(0, 256, 17)

TWO_ROWS_LINSPACE = np.vstack((LINSPACE * SCALE_1, LINSPACE * SCALE_2))

LINSPACE_INT4_ASYM = np.arange(0, 16)
TWO_ROWS_LINSPACE_INT4_ASYM = np.vstack((LINSPACE_INT4_ASYM * SCALE_1, LINSPACE_INT4_ASYM * SCALE_2))

LINSPACE_INT4_SYM = np.arange(-7, 8)
TWO_ROWS_LINSPACE_INT4_SYM = np.vstack((LINSPACE_INT4_SYM * SCALE_1, LINSPACE_INT4_SYM * SCALE_2))

TWO_OTHER_ROWS_LINSPACE_INT4_SYM = np.vstack((LINSPACE_INT4_SYM * SCALE_3, LINSPACE_INT4_SYM * SCALE_4))
TWO_GROUPS_IN_TWO_ROWS_SYM = np.hstack((TWO_ROWS_LINSPACE_INT4_SYM, TWO_OTHER_ROWS_LINSPACE_INT4_SYM))

TWO_OTHER_ROWS_LINSPACE_INT4_ASYM = np.vstack((LINSPACE_INT4_ASYM * SCALE_3, LINSPACE_INT4_ASYM * SCALE_4))
TWO_GROUPS_IN_TWO_ROWS_ASYM = np.hstack((TWO_ROWS_LINSPACE_INT4_ASYM, TWO_OTHER_ROWS_LINSPACE_INT4_ASYM))


int4_sym_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM, group_size=-1)
int4_asym_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_ASYM, group_size=-1)
int4_sym_grouped_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM, group_size=15)
int4_asym_grouped_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_ASYM, group_size=16)
LIST_DESCS = [
    # zero error
    QuantErrorDesc(name="2 rows of scaled [0, 255] linspace", weight=TWO_ROWS_LINSPACE),
    QuantErrorDesc(name="2 rows of scaled [-7, 7] linspace", weight=TWO_ROWS_LINSPACE_INT4_SYM, config=int4_sym_config),
    QuantErrorDesc(
        name="2 rows of scaled [0, 15] linspace", weight=TWO_ROWS_LINSPACE_INT4_ASYM, config=int4_asym_config
    ),
    QuantErrorDesc(
        name="two groups in two rows sym",
        weight=TWO_GROUPS_IN_TWO_ROWS_SYM,
        config=int4_sym_grouped_config,
    ),
    QuantErrorDesc(
        name="two groups in two rows asym",
        weight=TWO_GROUPS_IN_TWO_ROWS_ASYM,
        config=int4_asym_grouped_config,
    ),
    # non-zero error
    QuantErrorDesc(name="2 rows scaled [1, 254] linspace", weight=TWO_ROWS_LINSPACE[:, 1:-1], ref_error=239, atol=1),
    QuantErrorDesc(
        name="2 columns of scaled [0, 255] linspace", weight=np.transpose(TWO_ROWS_LINSPACE), ref_error=46818, atol=1
    ),
    QuantErrorDesc(
        name="2 rows of scaled [0, 15] linspace for sym",
        weight=TWO_ROWS_LINSPACE_INT4_ASYM,
        config=int4_sym_config,
        ref_error=4.12,
        atol=1,
    ),
    QuantErrorDesc(
        name="2 columns of of scaled [0, 15] linspace for sym",
        weight=np.transpose(TWO_ROWS_LINSPACE_INT4_ASYM),
        config=int4_sym_config,
        ref_error=5.87,
        atol=1,
    ),
    QuantErrorDesc(
        name="2 rows [1,14] linspace for asym",
        weight=TWO_ROWS_LINSPACE_INT4_ASYM[:, 1:-1],
        config=int4_asym_config,
        ref_error=1.49,
        atol=1,
    ),
    QuantErrorDesc(
        name="2 columns of [0-15] linspace for asym",
        weight=np.transpose(TWO_ROWS_LINSPACE_INT4_ASYM),
        config=int4_asym_config,
        ref_error=162,
        atol=1,
    ),
]


@pytest.mark.parametrize("desc", LIST_DESCS, ids=map(str, LIST_DESCS))
def test_quantization_error_calculation(desc: QuantErrorDesc):
    weight = desc.weight
    axis = (1,)
    actual_error = _get_integer_quantization_error(weight, axis, desc.config)
    ref_error = desc.ref_error
    atol = desc.atol if desc.atol is not None else 1e-8
    assert np.allclose(actual_error, ref_error, atol=atol)


WEIGHTS_2x4 = np.array([[-4, -3, -2, -1], [0, 11, 2, 3]])  # [2, 4]
WEIGHTS_abs_max = np.array([4, 2, 11, 3])  # [4]


@dataclass
class CalculateScaleDesc:
    weight: np.array
    ref_scale: np.array
    axis: int
    group_size: int


CALCULATE_SCALE_DESCS = [
    CalculateScaleDesc(weight=WEIGHTS_2x4, ref_scale=WEIGHTS_abs_max.reshape([2, 2, 1]), axis=1, group_size=2),
    CalculateScaleDesc(weight=WEIGHTS_2x4, ref_scale=np.abs(WEIGHTS_2x4).reshape([2, 1, 4]), axis=0, group_size=1),
    CalculateScaleDesc(
        weight=WEIGHTS_2x4.reshape([1, 2, 4, 1]),
        ref_scale=WEIGHTS_abs_max.reshape([1, 2, 2, 1, 1]),
        axis=2,
        group_size=2,
    ),
    CalculateScaleDesc(
        weight=WEIGHTS_2x4.reshape([1, 2, 4, 1]),
        ref_scale=np.abs(WEIGHTS_2x4.reshape([1, 2, 4, 1])),
        axis=0,
        group_size=1,
    ),
    CalculateScaleDesc(
        weight=WEIGHTS_2x4.reshape([2, 2, 2]), ref_scale=WEIGHTS_abs_max.reshape([2, 2, 1, 1]), axis=2, group_size=2
    ),
    CalculateScaleDesc(
        weight=WEIGHTS_2x4.reshape([2, 2, 2]),
        ref_scale=np.array([4, 3, 2, 11]).reshape([2, 1, 1, 2]),
        axis=1,
        group_size=2,
    ),
    CalculateScaleDesc(
        weight=WEIGHTS_2x4.reshape([2, 2, 2]),
        ref_scale=np.array([4, 11, 2, 3]).reshape([1, 1, 2, 2]),
        axis=0,
        group_size=2,
    ),
]


@pytest.mark.parametrize(
    ("ignored_scope", "num_compressed"),
    (
        (IgnoredScope(types=["MatMul"]), 1),
        (IgnoredScope(types=["Gather"]), 2),
        (IgnoredScope(names=["MatMul_1"]), 2),
        (IgnoredScope(patterns=["MatMul_\\d"]), 1),
    ),
)
def test_weight_compress_with_ignored_scope(ignored_scope, num_compressed):
    model = IntegerModel().ov_model
    compressed_model = compress_weights(model, ignored_scope=ignored_scope)
    ref_compressed_weights = TEST_MODELS[IntegerModel]
    act_num = 0
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            if op.get_element_type() == ov.Type(np.uint8):
                act_num += 1
    assert act_num == num_compressed


@pytest.mark.parametrize("desc", CALCULATE_SCALE_DESCS)
def test_calculate_scale_per_group(desc: CalculateScaleDesc):
    reshaped_weight, reduction_axis = _reshape_weights_for_grouped_quantization(
        desc.weight, reduction_axes=desc.axis, group_size=desc.group_size
    )
    act_scale = np.max(np.abs(reshaped_weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    assert np.allclose(act_scale, desc.ref_scale)


def test_raise_error_for_many_axes():
    with pytest.raises(RuntimeError):
        _reshape_weights_for_grouped_quantization(WEIGHTS_2x4, reduction_axes=(0, 1), group_size=1)


def test_raise_error_with_incorrect_group_size():
    with pytest.raises(RuntimeError):
        _reshape_weights_for_grouped_quantization(WEIGHTS_2x4, reduction_axes=(0,), group_size=3)


def test_raise_error_with_int8_and_non_default_ratio(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, ratio=0.5)


def test_raise_error_with_int8_and_non_default_group_size(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, group_size=64)
