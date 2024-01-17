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
from nncf import SensitivityMetric
from nncf.data.dataset import Dataset
from nncf.experimental.tensor import Tensor
from nncf.openvino.graph.node_utils import get_const_value
from nncf.quantization import compress_weights
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.scopes import IgnoredScope
from tests.openvino.native.models import GatherAndMatmulShareData
from tests.openvino.native.models import GatherWithTwoReductionAxes
from tests.openvino.native.models import IdentityMatmul
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

DATA_BASED_SENSITIVITY_METRICS = (
    SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
    SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
)

ALL_SENSITIVITY_METRICS = DATA_BASED_SENSITIVITY_METRICS + (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,)

INT8_MODES = (CompressWeightsMode.INT8, CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM)
INT4_NF4_MODES = (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM, CompressWeightsMode.NF4)


def get_next_node(node):
    target_inputs = node.output(0).get_target_inputs()
    assert len(target_inputs) == 1
    next_node = next(iter(target_inputs)).get_node()
    return next_node


def check_int8_node(op: ov.Node, mode: CompressWeightsMode = CompressWeightsMode.INT8_ASYM):
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
    if mode == CompressWeightsMode.INT8_SYM:
        assert list(zero_point_node.shape) == [1]
    else:
        reduced_weight_shape = list(op.shape)
        reduced_weight_shape[-1] = 1
        assert list(zero_point_node.shape) == reduced_weight_shape

    mul_node = get_next_node(sub_node)
    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    scale = get_const_value(scale_node)

    return {
        "compressed_weight": compressed_weight,
        "zero_point": zero_point,
        "scale": scale,
    }


def check_int4_grouped(op: ov.Node, mode: CompressWeightsMode, group_size: int = 7):
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


def check_nf4_grouped(op: ov.Node, group_size: int = 7):
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


def check_int8_sym(op: ov.Node):
    return check_int8_node(op, mode=CompressWeightsMode.INT8_SYM)


def get_mixed_mapping(primary_fn: Callable, list_layers: List[str]):
    mapping = {node_name: check_int8_node for node_name in list_layers}
    primary_node_name = TEST_MODELS[IntegerModel][0]
    mapping[primary_node_name] = primary_fn
    return mapping


@pytest.mark.parametrize(
    ("mode", "group_size", "check_fn_per_node_map"),
    (
        (CompressWeightsMode.INT8_ASYM, -1, {node_name: check_int8_node for node_name in TEST_MODELS[IntegerModel]}),
        (CompressWeightsMode.INT8_SYM, -1, {node_name: check_int8_sym for node_name in TEST_MODELS[IntegerModel]}),
        (CompressWeightsMode.INT4_SYM, 7, get_mixed_mapping(check_int4_sym_grouped, TEST_MODELS[IntegerModel])),
        (CompressWeightsMode.INT4_ASYM, 7, get_mixed_mapping(check_int4_asym_grouped, TEST_MODELS[IntegerModel])),
        (CompressWeightsMode.NF4, 7, get_mixed_mapping(check_nf4_grouped, TEST_MODELS[IntegerModel])),
    ),
)
def test_compare_compressed_weights(mode, group_size, check_fn_per_node_map):
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


@pytest.mark.parametrize(
    ("mode", "all_layers", "ratio", "ref_ids"),
    (
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 1, [0, 1, 2, 3, 4]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.8, [0, 3, 4]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.4, [0]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, True, 0.2, []),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 1, [0, 1, 2, 3]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.8, [0, 1, 3]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.4, [0]),
        (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR, False, 0.2, []),
        (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, True, 0.8, [0, 1, 2]),
        (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, False, 0.8, [0, 1, 2]),
        (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, True, 0.8, [0, 1, 2]),
        (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, False, 0.8, [0, 1, 2]),
        (SensitivityMetric.MAX_ACTIVATION_VARIANCE, True, 0.8, [0, 1, 2]),
        (SensitivityMetric.MAX_ACTIVATION_VARIANCE, False, 0.8, [0, 1, 2]),
        (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, True, 0.8, [0, 1, 2]),
        (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, False, 0.8, [0, 1, 2]),
    ),
)
def test_mixed_precision(mode, all_layers, ratio, ref_ids, mocker):
    model = SequentialMatmulModel().ov_model
    dataset = Dataset([np.ones([3, 3]), np.arange(9).reshape(3, 3)])
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.NF4,
        ratio=ratio,
        group_size=1,
        all_layers=all_layers,
        sensitivity_metric=mode,
        dataset=dataset,
    )
    names = {
        op.get_friendly_name() for op in compressed_model.get_ordered_ops() if op.get_element_type() == ov.Type.nf4
    }
    ref_nf4_nodes = {f"weights_{i}" for i in ref_ids}
    assert ref_nf4_nodes == names


@pytest.mark.parametrize("metric", DATA_BASED_SENSITIVITY_METRICS)
def test_gather_in_4_bit_if_all_layers_with_data(metric):
    model = IntegerModel().ov_model
    dataset = Dataset([np.arange(7).reshape(1, 7, 1)])
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.5,
        group_size=1,
        all_layers=True,
        sensitivity_metric=metric,
        dataset=dataset,
    )
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and "gather" in op.get_friendly_name():
            assert op.get_element_type() == ov.Type.u4


def test_gather_can_be_8_bit_if_all_layers_without_data():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.5,
        group_size=1,
        all_layers=True,
    )
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and "gather" in op.get_friendly_name():
            assert ov.Type(np.uint8) == op.get_element_type()


def test_gather_can_be_4_bit_if_all_layers_without_data():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1,
        group_size=1,
        all_layers=True,
    )
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and "gather" in op.get_friendly_name():
            assert ov.Type.u4 == op.get_element_type()


@pytest.mark.parametrize("metric", ALL_SENSITIVITY_METRICS)
def test_gather_in_8_bit_if_not_all_layers(metric):
    model = IntegerModel().ov_model
    dataset = Dataset([np.ones([1, 7, 1])])
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.5,
        group_size=1,
        all_layers=False,
        sensitivity_metric=metric,
        dataset=dataset,
    )
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and "gather" in op.get_friendly_name():
            assert op.get_element_type() == ov.Type(np.uint8)


MAX_BASELINE_SCORE = 1 / np.finfo(np.float32).eps
NON_ZERO_ROW = [-4, 1, 2]
ACTIVATION = np.array([NON_ZERO_ROW, [0, 0, 0], [0, 0, 0]])
MAX_VAR = 3.555555  # np.max(np.var(ACTIVATION, 0))
MEAN_VAR = 1.555555  # np.mean(np.var(ACTIVATION, 0))
MEAN_MAX = 2.333333  # np.mean(np.max(np.abs(ACTIVATION), 0))
HESSIAN_TRACE = (16 + 1 + 4) * 2 / 9  # sum(i*i for i in NON_ZERO_ROW) * 2 / ACTIVATION.size


@pytest.mark.parametrize(
    ("mode", "ref_act_scores", "ref_scores"),
    (
        (SensitivityMetric.HESSIAN_INPUT_ACTIVATION, HESSIAN_TRACE, 0),
        (SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE, MEAN_MAX, MEAN_MAX * MAX_BASELINE_SCORE),
        (SensitivityMetric.MEAN_ACTIVATION_VARIANCE, MEAN_VAR, MEAN_VAR * MAX_BASELINE_SCORE),
        (SensitivityMetric.MAX_ACTIVATION_VARIANCE, MAX_VAR, MAX_VAR * MAX_BASELINE_SCORE),
    ),
)
def test_data_based_criterion(mode, ref_scores, ref_act_scores, mocker):
    model = IdentityMatmul().ov_model
    dataset = Dataset([ACTIVATION])
    criterion_cls = MIXED_PRECISION_CRITERIA.get(mode)
    scores_spy = mocker.spy(criterion_cls, "_calc_sensitivity")
    act_scores_spy = mocker.spy(criterion_cls, "_calc_activation_sensitivity")

    compress_weights(
        model,
        mode=CompressWeightsMode.NF4,
        ratio=0.5,
        group_size=1,
        dataset=dataset,
        sensitivity_metric=mode,
        all_layers=True,
    )
    scores = scores_spy.spy_return
    act_scores = act_scores_spy.spy_return
    assert np.allclose(scores, ref_scores)
    assert np.allclose(act_scores, ref_act_scores)


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM))
def test_not_quantize_with_multiple_reduction_axes(mode):
    model = GatherWithTwoReductionAxes().ov_model
    compressed_model = compress_weights(model, mode=mode)
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() == "gather_1_data":
            assert op.get_element_type() == ov.Type(np.float32)


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM))
def test_shared_gather(mode):
    weight_name_vs_type = {
        "gather_2_data": ov.Type(np.uint8),
        "shared_data": ov.Type(np.uint8),
        "matmul_1_data": ov.Type.u4,
    }
    model = GatherAndMatmulShareData().ov_model
    compressed_model = compress_weights(model, mode, group_size=3)
    for op in compressed_model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and op_name in weight_name_vs_type:
            assert op.get_element_type() == weight_name_vs_type[op_name]


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
    weight = Tensor(desc.weight)
    axis = 1
    actual_error = get_integer_quantization_error(weight, axis, desc.config)
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
        if (
            op.get_type_name() == "Constant"
            and op.get_friendly_name() in ref_compressed_weights
            and op.get_element_type() == ov.Type(np.uint8)
        ):
            act_num += 1
    assert act_num == num_compressed


@pytest.mark.parametrize("desc", CALCULATE_SCALE_DESCS)
def test_calculate_scale_per_group(desc: CalculateScaleDesc):
    reshaped_weight, reduction_axis = reshape_weight_for_grouped_quantization(
        desc.weight, reduction_axis=desc.axis, group_size=desc.group_size
    )
    act_scale = np.max(np.abs(reshaped_weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    assert np.allclose(act_scale, desc.ref_scale)


def test_raise_error_for_many_axes():
    with pytest.raises(AssertionError):
        reshape_weight_for_grouped_quantization(WEIGHTS_2x4, reduction_axis=(0, 1), group_size=1)


def test_raise_error_with_tuple():
    with pytest.raises(AssertionError):
        reshape_weight_for_grouped_quantization(WEIGHTS_2x4, reduction_axis=(0,), group_size=3)


@pytest.mark.parametrize("mode", INT8_MODES)
@pytest.mark.parametrize(
    "params",
    (
        {"ratio": 0.5},
        {"group_size": 64},
        {"all_layers": True},
        {"all_layers": False},
        *({"sensitivity_metric": metric} for metric in ALL_SENSITIVITY_METRICS),
        {"dataset": "anything"},
    ),
)
def test_raise_error_with_unsupported_params_for_int8(mocker, mode, params):
    with pytest.raises(AttributeError):
        compress_weights(ov.Model([], []), mode=mode, **params)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
@pytest.mark.parametrize("metric", DATA_BASED_SENSITIVITY_METRICS)
def test_raise_error_with_data_metric_and_without_dataset(mode, metric):
    model = IntegerModel().ov_model
    with pytest.raises(AttributeError):
        compress_weights(model, mode=mode, sensitivity_metric=metric, group_size=-1, ratio=0.8)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_by_default(mocker, mode):
    model = IntegerModel().ov_model
    dataset = Dataset([np.ones([1, 7, 1])])
    criterion_cls = MIXED_PRECISION_CRITERIA.get(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
    scores_spy = mocker.spy(criterion_cls, "_calc_sensitivity")

    compress_weights(model, mode=mode, ratio=0.8, group_size=-1, dataset=dataset)

    scores_spy.assert_called()
