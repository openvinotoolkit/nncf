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

from functools import partial
from typing import List

import numpy as np
import openvino.runtime as ov
import pytest
from attr import dataclass

from nncf import CompressWeightsMode
from nncf.openvino.graph.node_utils import get_const_value
from nncf.quantization import compress_weights
from nncf.quantization.algorithms.weight_compression.openvino_backend import _calculate_scale_per_group
from nncf.quantization.algorithms.weight_compression.openvino_backend import _get_int8_err
from nncf.quantization.algorithms.weight_compression.openvino_backend import _get_nf4_error
from nncf.scopes import IgnoredScope
from tests.openvino.native.models import IntegerModel
from tests.openvino.native.models import SequentialMatmulModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.quantization.test_fq_params_calculation import REFERENCE_SCALES_DIR
from tests.shared.helpers import compare_stats
from tests.shared.helpers import load_json

TEST_MODELS = {
    IntegerModel: ["gather_2_data", "matmul_1_data", "matmul_2_data"],
    WeightsModel: ["weights_0", "weights_1"],
}


@pytest.mark.parametrize("model_creator_func", TEST_MODELS)
def test_compress_weights_int8(model_creator_func, tmp_path):
    ref_compressed_weights = TEST_MODELS[model_creator_func]
    name = model_creator_func().__class__.__name__
    model = model_creator_func().ov_model
    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            assert op.get_element_type() == ov.Type(np.uint8)
            n_compressed_weights += 1
    ov.serialize(compressed_model, tmp_path / (name + ".xml"))
    assert n_compressed_weights == len(ref_compressed_weights)


@pytest.mark.parametrize("model_creator_func", TEST_MODELS)
def test_compress_weights_nf4(model_creator_func):
    if issubclass(IntegerModel, model_creator_func):
        pytest.xfail("Waiting for the merge NF4 support in OV - PR 19900")
    ref_compressed_weights = TEST_MODELS[model_creator_func]
    model = model_creator_func().ov_model
    compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4, ratio=1, group_size=1)

    n_compressed_weights = 0
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            if n_compressed_weights in (0, len(ref_compressed_weights) - 1):
                assert op.get_element_type() == ov.Type(np.uint8)
            else:
                assert op.get_element_type() == ov.Type.nf4

            n_compressed_weights += 1

    assert n_compressed_weights == len(ref_compressed_weights)


def get_next_node(node):
    target_inputs = node.output(0).get_target_inputs()
    assert len(target_inputs) == 1
    next_node = next(iter(target_inputs)).get_node()
    return next_node


def test_compare_compressed_weights():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(model)
    nodes = {}
    ref_compressed_weights = TEST_MODELS[IntegerModel]
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_compressed_weights:
            assert op.get_element_type() == ov.Type(np.uint8)
            compressed_weight = get_const_value(op)

            convert_node = get_next_node(op)
            assert convert_node.get_type_name() == "Convert"

            sub_node = get_next_node(convert_node)
            assert sub_node.get_type_name() == "Subtract"
            zero_point_node = sub_node.input_value(1).get_node()
            zero_point = get_const_value(zero_point_node)

            mul_node = get_next_node(sub_node)
            assert mul_node.get_type_name() == "Multiply"
            scale_node = mul_node.input_value(1).get_node()
            scale = get_const_value(scale_node)

            nodes[op.get_friendly_name()] = {
                "compressed_weight": compressed_weight,
                "zero_point": zero_point,
                "scale": scale,
            }

    ref_stats_path = REFERENCE_SCALES_DIR / "IntegerModel_compressed_weights.json"

    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    params = ["compressed_weight", "zero_point", "scale"]
    compare_stats(ref_nodes, nodes, params)


# TODO(nlyalyus) Waiting for the merge NF4 support in OV - PR 19900
@pytest.mark.xfail
def test_compare_compressed_weights_nf4():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(model, mode=CompressWeightsMode.NF4, ratio=1, group_size=3)

    nodes = {}
    ref_nf4_weight = TEST_MODELS[IntegerModel][1]
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() in ref_nf4_weight:
            assert op.get_element_type() == ov.Type.nf4
            # TODO: should be fixed in python api
            with pytest.raises(RuntimeError):
                get_const_value(op)

            convert_node = get_next_node(op)
            assert convert_node.get_type_name() == "Convert"

            mul_node = get_next_node(convert_node)
            assert mul_node.get_type_name() == "Multiply"
            scale_node = mul_node.input_value(1).get_node()
            scale = get_const_value(scale_node)

            reshape_node = get_next_node(mul_node)
            assert reshape_node.get_type_name() == "Reshape"

            nodes[op.get_friendly_name()] = {
                # "compressed_weight": compressed_weight,
                "scale": scale,
            }

    ref_stats_path = REFERENCE_SCALES_DIR / "IntegerModel_compressed_weights_nf4.json"

    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    params = [
        # "compressed_weight",
        "scale"
    ]
    compare_stats(ref_nodes, nodes, params)


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
class BaseDesc:
    weight: List[float]
    ref_error: int = 0
    axis = (1,)
    name: str = ""
    atol: float = None

    def get_error_fn(self) -> float:
        raise NotImplementedError

    def __str__(self):
        prefix = "exact_match_" if self.ref_error == 0 else ""
        name = self.name.replace(" ", "_") if self.name else self.__class__.__name__
        return prefix + name


@dataclass
class Int8Desc(BaseDesc):
    def get_error_fn(self) -> float:
        return partial(_get_int8_err, reduction_axes=self.axis)

    def __str__(self):
        base_str = super().__str__()
        return "int8_" + base_str


@dataclass
class NF4Desc(BaseDesc):
    group_size: int = -1

    def get_error_fn(self) -> float:
        return partial(_get_nf4_error, reduction_axes=self.axis, group_size=self.group_size)

    def __str__(self):
        base_str = super().__str__()
        return "nf4_" + base_str


SCALE_1 = 1.2
SCALE_2 = 3.4
SCALE_3 = 5.6
SCALE_4 = 7.8
LINSPACE = np.arange(0, 256, 17)
NF4_LOOKUP = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)

TWO_ROWS_NF4 = np.vstack((NF4_LOOKUP * SCALE_1, NF4_LOOKUP * SCALE_2))
TWO_OTHER_ROWS_NF4 = np.vstack((NF4_LOOKUP * SCALE_3, NF4_LOOKUP * SCALE_4))
TWO_ROWS_LINSPACE = np.vstack((LINSPACE * SCALE_1, LINSPACE * SCALE_2))
TWO_GROUPS_IN_TWO_ROWS_NF4 = np.hstack((TWO_ROWS_NF4, TWO_OTHER_ROWS_NF4))
TWO_GROUPS_IN_TWO_ROWS_NO_1_NF4 = np.hstack((TWO_ROWS_NF4[:, 1:-1], TWO_OTHER_ROWS_NF4[:, 1:-1]))

LIST_DESCS = [
    # zero error
    Int8Desc(name="2 rows of 0-255 linspace", weight=TWO_ROWS_LINSPACE),
    NF4Desc(name="2 rows of exact quantiles", weight=TWO_ROWS_NF4),
    NF4Desc(name="two groups in two rows", weight=TWO_GROUPS_IN_TWO_ROWS_NF4, group_size=16),
    # non-zero error
    Int8Desc(name="2 rows 1-254 linspace", weight=TWO_ROWS_LINSPACE[:, 1:-1], ref_error=239, atol=1),
    Int8Desc(name="2 columns of 0-255 linspace", weight=np.transpose(TWO_ROWS_LINSPACE), ref_error=46818, atol=1),
    NF4Desc(name="2 rows of exact quantiles without -1 and 1", weight=TWO_ROWS_NF4[:, 1:-1], ref_error=5e-3, atol=1e-3),
    NF4Desc(name="2 columns of exact quantiles", weight=np.transpose(TWO_ROWS_NF4), ref_error=1e-2, atol=1e-2),
    NF4Desc(
        name="two groups in two rows without -1 and 1",
        weight=TWO_GROUPS_IN_TWO_ROWS_NO_1_NF4,
        group_size=14,
        ref_error=2e-2,
        atol=1e-2,
    ),
]


@pytest.mark.parametrize("desc", LIST_DESCS, ids=map(str, LIST_DESCS))
def test_quantization_error_calculation(desc: BaseDesc):
    weight = desc.weight
    actual_error = desc.get_error_fn()(weight)
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
    act_scale, _ = _calculate_scale_per_group(desc.weight, reduction_axes=desc.axis, group_size=desc.group_size)
    assert np.allclose(act_scale, desc.ref_scale)


def test_raise_error_for_many_axes():
    with pytest.raises(RuntimeError):
        _calculate_scale_per_group(WEIGHTS_2x4, reduction_axes=(0, 1), group_size=1)


def test_raise_error_with_incorrect_group_size():
    with pytest.raises(RuntimeError):
        _calculate_scale_per_group(WEIGHTS_2x4, reduction_axes=(0,), group_size=3)


def test_raise_error_with_int8_and_non_default_ratio(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, ratio=0.5)


def test_raise_error_with_int8_and_non_default_group_size(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, group_size=64)
