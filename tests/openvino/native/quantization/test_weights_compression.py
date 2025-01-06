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

import inspect
import os
from typing import Callable, List

import numpy as np
import openvino.runtime as ov
import pandas as pd
import pytest
from attr import dataclass
from openvino.runtime import opset13 as opset

import nncf
from nncf import CompressWeightsMode
from nncf import SensitivityMetric
from nncf.common.utils.debug import nncf_debug
from nncf.data.dataset import Dataset
from nncf.experimental.common.tensor_statistics.collectors import AggregatorBase
from nncf.openvino.graph.node_utils import get_const_value
from nncf.parameters import BackupMode
from nncf.quantization import compress_weights
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters as CompressionParams
from nncf.quantization.advanced_parameters import AdvancedGPTQParameters as GPTQParams
from nncf.quantization.advanced_parameters import AdvancedLoraCorrectionParameters as LoraParams
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.scopes import IgnoredScope
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from tests.cross_fw.shared.comparator import compare_stats
from tests.cross_fw.shared.json import dump_to_json
from tests.cross_fw.shared.json import load_json
from tests.openvino.native.common import get_actual_reference_for_current_openvino
from tests.openvino.native.models import AWQActMatmulModel
from tests.openvino.native.models import AWQMatmulModel
from tests.openvino.native.models import GatherAndMatmulShareData
from tests.openvino.native.models import GatherWithTwoReductionAxes
from tests.openvino.native.models import IdentityMatmul
from tests.openvino.native.models import IntegerModel
from tests.openvino.native.models import ModelNamedConsts
from tests.openvino.native.models import OVReferenceModel
from tests.openvino.native.models import SequentialMatmulModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.quantization.test_fq_params_calculation import REFERENCE_SCALES_DIR

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
INT4_MODES = (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM)


class LMLinearModel(OVReferenceModel):
    OUTPUT_DIM = 32
    HIDDEN_DIM = 16
    INPUT_SHAPE = [1, 24, HIDDEN_DIM]  # [B, SeqLen, HiddenDim]

    def _create_ov_model(self, transpose_b: bool = True, transpose_a=False, input_shape=None):
        self._input_shape = self.INPUT_SHAPE if input_shape is None else input_shape
        hdim_axis = -2 if transpose_a else -1
        self._hidden_dim = self._input_shape[hdim_axis]
        input_1 = opset.parameter(self._input_shape, name="Input")
        weight_shape = self.get_weight_shape(transpose_b)
        data = self._rng.random(weight_shape).astype(np.float32)
        matmul = opset.matmul(input_1, data, transpose_a=transpose_a, transpose_b=transpose_b, name="MatMul")
        result = opset.result(matmul, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_weight_shape(self, transpose_b: bool = True):
        return [self.OUTPUT_DIM, self.hidden_dim] if transpose_b else [self.hidden_dim, self.OUTPUT_DIM]


def get_next_node(node):
    target_inputs = node.output(0).get_target_inputs()
    assert len(target_inputs) == 1
    next_node = next(iter(target_inputs)).get_node()
    return next_node


def check_int8_node(op: ov.Node, mode: CompressWeightsMode = CompressWeightsMode.INT8_ASYM):
    dtype = ov.Type.u8 if mode == CompressWeightsMode.INT8_ASYM else ov.Type.i8
    assert op.get_element_type() == dtype
    compressed_weight = get_const_value(op)
    stats = {"compressed_weight": compressed_weight}

    convert_node = get_next_node(op)
    assert convert_node.get_type_name() == "Convert"

    if mode == CompressWeightsMode.INT8_ASYM:
        sub_node = get_next_node(convert_node)
        assert sub_node.get_type_name() == "Subtract"

        convert_node = sub_node.input_value(1).get_node()
        assert convert_node.get_type_name() == "Convert"

        zero_point_node = convert_node.input_value(0).get_node()
        zero_point = get_const_value(zero_point_node)
        stats["zero_point"] = zero_point
        reduced_weight_shape = list(op.shape)
        reduced_weight_shape[-1] = 1
        assert list(zero_point_node.shape) == reduced_weight_shape
        mul_node = get_next_node(sub_node)
    else:
        mul_node = get_next_node(convert_node)

    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    scale = get_const_value(scale_node)
    stats["scale"] = scale
    return stats


def check_int4_grouped(op: ov.Node, mode: CompressWeightsMode, group_size: int = 7):
    dtype = ov.Type.u4 if mode == CompressWeightsMode.INT4_ASYM else ov.Type.i4
    assert op.get_element_type() == dtype
    weight_shape = op.shape
    # NOTE: get_const_value doesn't work for 4-bit types
    assert list(weight_shape)[-1] == group_size
    reduced_weight_shape = list(weight_shape)
    reduced_weight_shape[-1] = 1

    convert_node = get_next_node(op)
    assert convert_node.get_type_name() == "Convert"

    if mode == CompressWeightsMode.INT4_ASYM:
        sub_node = get_next_node(convert_node)
        assert sub_node.get_type_name() == "Subtract"

        convert_node = sub_node.input_value(1).get_node()
        assert convert_node.get_type_name() == "Convert"

        zero_point_node = convert_node.input_value(0).get_node()
        assert zero_point_node.get_element_type() == dtype
        assert list(zero_point_node.shape) == reduced_weight_shape
        mul_node = get_next_node(sub_node)
    else:
        mul_node = get_next_node(convert_node)

    assert mul_node.get_type_name() == "Multiply"
    scale_node = mul_node.input_value(1).get_node()
    assert list(scale_node.shape) == reduced_weight_shape

    reshape_node = get_next_node(mul_node)
    assert reshape_node.get_type_name() == "Reshape"

    convert_node = get_next_node(reshape_node)
    assert convert_node.get_type_name() == "Convert"

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

    convert_node = get_next_node(reshape_node)
    assert convert_node.get_type_name() == "Convert"

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

    ref_stats_path = get_actual_reference_for_current_openvino(
        REFERENCE_SCALES_DIR / f"IntegerModel_compressed_weights_{mode.value}.json"
    )

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
    dataset = Dataset([np.ones([1, 3, 3]), np.arange(9).reshape(1, 3, 3)])
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
    dim1 = 2  # sequence length dimension
    dim2 = 7
    max_input_value = 6
    model = IntegerModel(dim1=dim1, dim2=dim2, max_input_value=max_input_value, add_batch_dimension=True).ov_model

    input_shape = (dim1, dim2, dim1)
    n_inputs = input_shape[0] * input_shape[1] * input_shape[2]
    n_copies = int(np.ceil(n_inputs / (max_input_value + 1)))
    # Rolling is needed to ensure non-zero variance
    input_ = [np.roll(np.arange(max_input_value + 1), i + 1) for i in range(n_copies)]
    input_ = np.hstack(input_)[:n_inputs]
    input_ = input_.reshape(input_shape)

    dataset = Dataset([input_])
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.5,
        group_size=1,
        all_layers=True,
        sensitivity_metric=metric,
        dataset=dataset,
    )
    int4_reference_node_names = ["gather_2_data"]
    nodes_map = {op.get_friendly_name(): op for op in compressed_model.get_ordered_ops()}
    for node_name in int4_reference_node_names:
        node = nodes_map[node_name]
        assert node.get_type_name() == "Constant"
        assert node.get_element_type() == ov.Type.i4


def test_gather_can_be_8_bit_if_all_layers_without_data():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=0.5,
        group_size=1,
        all_layers=True,
    )
    int8_reference_node_names = ["gather_2_data", "gather_2_data/zero_point"]
    nodes_map = {op.get_friendly_name(): op for op in compressed_model.get_ordered_ops()}
    for node_name in int8_reference_node_names:
        node = nodes_map[node_name]
        assert node.get_type_name() == "Constant"
        assert node.get_element_type() == ov.Type.u8


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM))
def test_conv_in_8_bit_if_mode_8bit(mode):
    model = WeightsModel().ov_model
    compressed_model = compress_weights(model, mode=mode)
    int8_reference_node_names = ["conv_weights_0", "conv_weights_1"]
    nodes_map = {op.get_friendly_name(): op for op in compressed_model.get_ordered_ops()}
    dtype = ov.Type.u8 if mode == CompressWeightsMode.INT8_ASYM else ov.Type.i8
    for node_name in int8_reference_node_names:
        node = nodes_map[node_name]
        assert node.get_type_name() == "Constant"
        assert node.get_element_type() == dtype


@pytest.mark.parametrize("all_layers", (True, False))
def test_conv_in_8_bit_if_mode_4bit(all_layers):
    model = WeightsModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1,
        group_size=1,
        all_layers=all_layers,
    )
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant":
            if op.get_friendly_name() in [
                "conv_weights_0",
                "conv_weights_0/zero_point",
                "conv_weights_1",
                "conv_weights_1/zero_point",
            ]:
                assert ov.Type.u8 == op.get_element_type()
            elif op.get_friendly_name() in ["weights_1", "weights_1/zero_point"]:
                assert ov.Type.i4 == op.get_element_type()
            elif op.get_friendly_name() in ["weights_0", "weights_0/zero_point"]:
                dtype = ov.Type.i4 if all_layers else ov.Type.u8
                assert dtype == op.get_element_type()


def test_gather_can_be_4_bit_if_all_layers_without_data():
    model = IntegerModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1,
        group_size=1,
        all_layers=True,
    )
    int4_reference_node_names = ["gather_2_data"]
    nodes_map = {op.get_friendly_name(): op for op in compressed_model.get_ordered_ops()}
    for node_name in int4_reference_node_names:
        node = nodes_map[node_name]
        assert node.get_type_name() == "Constant"
        assert node.get_element_type() == ov.Type.i4


@pytest.mark.parametrize("metric", ALL_SENSITIVITY_METRICS)
def test_gather_in_8_bit_if_not_all_layers(metric):
    model = IntegerModel(add_batch_dimension=True).ov_model
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
    int8_reference_node_names = ["gather_2_data", "gather_2_data/zero_point"]
    nodes_map = {op.get_friendly_name(): op for op in compressed_model.get_ordered_ops()}
    for node_name in int8_reference_node_names:
        node = nodes_map[node_name]
        assert node.get_type_name() == "Constant"
        assert node.get_element_type() == ov.Type.u8


MAX_BASELINE_SCORE = 1 / np.finfo(np.float32).eps
NON_ZERO_ROW = [-4, 1, 2]
ACTIVATION = np.array([[NON_ZERO_ROW, [0, 0, 0], [0, 0, 0]]])
MAX_VAR = 3.555555  # np.max(np.var(ACTIVATION, 1))
MEAN_VAR = 1.555555  # np.mean(np.var(ACTIVATION, 1))
MEAN_MAX = 2.333333  # np.mean(np.max(np.abs(ACTIVATION), 1))
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
def test_quantize_Gather_with_multiple_reduction_axes_in_8bit(mode):
    model = GatherWithTwoReductionAxes().ov_model
    compressed_model = compress_weights(model, mode=mode)
    dtype = ov.Type.u8 if mode == CompressWeightsMode.INT8_ASYM else ov.Type.i8
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() == "gather_1_data":
            assert op.get_element_type() == dtype


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM))
@pytest.mark.parametrize("all_layers", (True, False))
def test_quantize_Gather_with_multiple_reduction_axes_if_mode_4bit(mode, all_layers):
    model = GatherWithTwoReductionAxes().ov_model
    compressed_model = compress_weights(model, mode=mode, all_layers=all_layers)
    for op in compressed_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name() == "gather_1_data":
            assert op.get_element_type() == ov.Type.u8


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM))
def test_shared_gather(mode):
    weight_name_vs_type = {
        "gather_2_data": ov.Type.u8,
        "shared_data": ov.Type.u8,
        "matmul_1_data": ov.Type.i4 if mode == CompressWeightsMode.INT4_SYM else ov.Type.u4,
    }
    model = GatherAndMatmulShareData().ov_model
    compressed_model = compress_weights(model, mode, group_size=3)
    for op in compressed_model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and op_name in weight_name_vs_type:
            assert op.get_element_type() == weight_name_vs_type[op_name]


@pytest.mark.parametrize("all_layers", (True, False))
def test_shared_gather_all_layers(all_layers):
    weight_name_vs_type = {
        "gather_2_data": ov.Type.u4 if all_layers else ov.Type.u8,
        "shared_data": ov.Type.u4 if all_layers else ov.Type.u8,
        "matmul_1_data": ov.Type.u4,
    }
    model = GatherAndMatmulShareData().ov_model
    compressed_model = compress_weights(model, CompressWeightsMode.INT4_ASYM, group_size=-1, all_layers=all_layers)
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

LINSPACE_INT4_SYM = np.arange(-8, 7)
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
        ref_error=0.63,
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
            and op.get_element_type() == ov.Type.u8
        ):
            act_num += 1
    assert act_num == num_compressed


@pytest.mark.parametrize("desc", CALCULATE_SCALE_DESCS)
def test_calculate_scale_per_group(desc: CalculateScaleDesc):
    reshaped_weight, reduction_axis = reshape_weight_for_grouped_quantization(
        desc.weight, reduction_axes=desc.axis, group_size=desc.group_size
    )
    act_scale = np.max(np.abs(reshaped_weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    assert np.allclose(act_scale, desc.ref_scale)


def test_raise_error_for_many_axes():
    with pytest.raises(nncf.UnsupportedModelError):
        reshape_weight_for_grouped_quantization(WEIGHTS_2x4, reduction_axes=(0, 1), group_size=1)


def test_raise_error_channel_size_is_not_divisible_by_group_size():
    with pytest.raises(nncf.UnsupportedModelError):
        reshape_weight_for_grouped_quantization(WEIGHTS_2x4, reduction_axes=(0,), group_size=3)


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
        {"scale_estimation": True},
        {"lora_correction": True},
        {"gptq": True},
        {"awq": True},
        {"backup_mode": BackupMode.NONE},
        {"backup_mode": BackupMode.INT8_ASYM},
        {"backup_mode": BackupMode.INT8_SYM},
        {"advanced_parameters": AdvancedCompressionParameters(statistics_path="anything")},
    ),
)
def test_raise_error_with_unsupported_params_for_int8(mode, params):
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(ov.Model([], []), mode=mode, **params)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
@pytest.mark.parametrize(
    "params",
    ({"dataset": "anything", "lora_correction": True, "gptq": True},),
)
def test_raise_error_with_unsupported_params_for_int4(mode, params):
    with pytest.raises(nncf.ValidationError):
        compress_weights(ov.Model([], []), mode=mode, **params)


@pytest.mark.parametrize(
    "algo",
    (
        "lora_correction",
        "awq",
        "scale_estimation",
        "gptq",
    ),
)
def test_raise_error_with_unsupported_params_for_e2m1(algo):
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(ov.Model([], []), dataset="anything", mode=CompressWeightsMode.E2M1, **{algo: True})


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
@pytest.mark.parametrize(
    "algo",
    (
        "lora_correction",
        "awq",
        "scale_estimation",
        "gptq",
    ),
)
def test_raise_error_with_unsupported_params_for_empty_dataset(mode, algo):
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(ov.Model([], []), dataset=None, mode=mode, **{algo: True})


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
@pytest.mark.parametrize("metric", DATA_BASED_SENSITIVITY_METRICS)
def test_raise_error_with_data_metric_and_without_dataset(mode, metric):
    model = IntegerModel().ov_model
    with pytest.raises(nncf.ValidationError):
        compress_weights(model, mode=mode, sensitivity_metric=metric, group_size=-1, ratio=0.8)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_by_default(mocker, mode):
    model = IntegerModel(add_batch_dimension=True).ov_model
    dataset = Dataset([np.ones([1, 7, 1])])
    criterion_cls = MIXED_PRECISION_CRITERIA.get(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
    scores_spy = mocker.spy(criterion_cls, "_calc_sensitivity")

    compress_weights(model, mode=mode, ratio=0.8, group_size=-1, dataset=dataset)

    scores_spy.assert_called()


@pytest.mark.parametrize("mode", INT4_MODES)
def test_call_max_var_criterion_with_dataset_by_default_awq(mode):
    model = AWQMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, awq=True)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
@pytest.mark.parametrize("with_multiply", (True, False))
def test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul(mode, with_multiply):
    n_layers = 8
    n_awq_target = n_layers - 1  # first MatMul is always int8
    model = AWQActMatmulModel(with_multiply=with_multiply, n_layers=n_layers).ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, awq=True)

    awq_num = 0
    for op in model.get_ops():
        if op.get_type_name() == "Constant" and "awq" in op.get_friendly_name():
            awq_num += 1
    assert awq_num == n_awq_target


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_awq_for_compressed_model(mode):
    model = AWQMatmulModel(is_int8=True).ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, awq=True)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_awq_neg_group_size(mode):
    model = AWQMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 8, 8])])
    compress_weights(model, mode=mode, ratio=1.0, group_size=-1, dataset=dataset, awq=True)


def test_data_type_for_num_weights(mocker):
    stub = mocker.stub()
    params = WeightCompressionParameters(stub, stub, stub, np.int32(1), stub)
    assert isinstance(params.num_weights, np.uint64)


def check_compressed_matmul_subgraph(start_node, activation_dtype, weight_dtype, is_adapter=False):
    # Weight scale should be in fp16 nevertheless the weight data type
    assert start_node.input_value(1).get_node().get_element_type() == ov.Type.f16

    next_node = start_node
    if not is_adapter:
        # lora adapters are int8 - no reshape for group quantization, it's only for int4 weights
        next_node = get_next_node(start_node)
        assert next_node.get_type_name() == "Reshape"

    next_node = get_next_node(next_node)
    if activation_dtype == ov.Type.f16:
        # There should be no convert node after multiply if both weights and activations are in f16
        assert next_node.get_type_name() != "Convert"
    else:
        assert next_node.get_type_name() == "Convert"
        # In case precision of weight and activation were equal, but not f16, the convert node is manually inserted
        # In case of lora adapter the convert is always manually inserted
        if (activation_dtype == weight_dtype and weight_dtype != ov.Type.f16) or is_adapter:
            ref_name = start_node.get_friendly_name() + "/convert"
            assert next_node.get_friendly_name() == ref_name


@pytest.mark.parametrize(
    "activation_dtype, weight_dtype",
    [
        (ov.Type.f32, ov.Type.f32),
        (ov.Type.f32, ov.Type.f16),
        (ov.Type.f32, ov.Type.bf16),
        (ov.Type.f16, ov.Type.f16),
        (ov.Type.bf16, ov.Type.bf16),
    ],
)
def test_compression_for_different_dtypes(activation_dtype, weight_dtype):
    model = IdentityMatmul(weights_dtype=weight_dtype, activation_dtype=activation_dtype).ov_model

    compressed_model = compress_weights(
        model, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=1, all_layers=True
    )

    name_to_node_map = {op.get_friendly_name(): op for op in compressed_model.get_ops()}
    scale_multiply_node = name_to_node_map["weights/fq_weights_1"]
    check_compressed_matmul_subgraph(scale_multiply_node, activation_dtype, weight_dtype)


DATASET_SIZE = 5


@pytest.mark.parametrize(
    ("dataset_size", "subset_size", "ref_size"),
    (
        (DATASET_SIZE, 1, 1),
        (DATASET_SIZE, DATASET_SIZE, DATASET_SIZE),
        (DATASET_SIZE, DATASET_SIZE + 1, DATASET_SIZE),
    ),
)
@pytest.mark.parametrize(
    ("compression_args", "multiplier_of_calls"),
    [
        ({"mode": CompressWeightsMode.INT4_ASYM, "ratio": 1}, 0),  # data-free, no reducers
        ({"mode": CompressWeightsMode.INT4_ASYM, "ratio": 1, "awq": True}, 2),  # mean & shape reducer for AWQ
        (
            {"mode": CompressWeightsMode.INT4_ASYM, "ratio": 0.5, "awq": True},
            3,
        ),  # 2 - for AWQ + 1 - for Mixed Precision
        (
            {
                "mode": CompressWeightsMode.INT4_ASYM,
                "ratio": 0.5,
                "sensitivity_metric": nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
            },
            1,
        ),  # 1 reducer for mixed precision
        (
            {
                "mode": CompressWeightsMode.INT4_ASYM,
                "ratio": 0.5,
                "sensitivity_metric": nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
            },
            1,
        ),  # 1 reducer for mixed precision
        (
            {
                "mode": CompressWeightsMode.INT4_ASYM,
                "ratio": 0.5,
                "sensitivity_metric": nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE,
            },
            1,
        ),  # 1 reducer for mixed precision
        (
            {
                "mode": CompressWeightsMode.INT4_ASYM,
                "ratio": 0.5,
                "sensitivity_metric": nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
            },
            1,
        ),  # 1 reducer for mixed precision
        (
            {
                "mode": CompressWeightsMode.INT4_ASYM,
                "ratio": 0.5,
                "sensitivity_metric": nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            },
            0,
        ),  # 0 - data-free method
    ],
)
def test_number_of_reduced_statistics_for_subset_size(
    mocker, dataset_size, subset_size, ref_size, compression_args, multiplier_of_calls
):
    model = IdentityMatmul().ov_model
    dataset = Dataset([ACTIVATION] * dataset_size)
    stats_spy = mocker.spy(AggregatorBase, "register_reduced_input")

    compress_weights(model, dataset=dataset, subset_size=subset_size, **compression_args)

    assert stats_spy.call_count == ref_size * multiplier_of_calls


def test_default_subset_value():
    default_value = inspect.signature(compress_weights).parameters["subset_size"].default
    assert default_value == 128


@pytest.mark.parametrize("subset_size", (-1, 0))
def test_invalid_subset_size(subset_size):
    model = IdentityMatmul().ov_model
    dataset = Dataset([ACTIVATION])
    with pytest.raises(nncf.ValidationError):
        compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, ratio=0.5, dataset=dataset, subset_size=subset_size)


def test_duplicate_names_generation():
    model = ModelNamedConsts().ov_model
    compressed_model = compress_weights(model)
    op_names = set()
    for op in compressed_model.get_ops():
        name = op.get_friendly_name()
        assert name not in op_names
        op_names.add(name)


@pytest.mark.parametrize(
    ("mode", "compressed_weight_dtype"),
    (
        (CompressWeightsMode.INT4_SYM, TensorDataType.int8),
        (CompressWeightsMode.INT4_ASYM, TensorDataType.uint8),
        (CompressWeightsMode.NF4, TensorDataType.float32),
    ),
)
def test_call_max_var_criterion_with_dataset_by_default_scale_estimation(mode, compressed_weight_dtype, mocker):
    model = AWQMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 8, 8])])
    from nncf.quantization.algorithms.weight_compression import scale_estimation
    from nncf.quantization.algorithms.weight_compression.algorithm import ScaleEstimation

    se_spy = mocker.spy(ScaleEstimation, "apply")
    tzm_spy = mocker.spy(scale_estimation, "get_target_zero_mask")

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, scale_estimation=True)

    assert se_spy.call_count == 1
    assert tzm_spy.call_args_list[0][0][0].dtype == compressed_weight_dtype


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_scale_estimation_for_compressed_model(mode):
    model = AWQMatmulModel(is_int8=True).ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, scale_estimation=True)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_scale_estimation_neg_group_size(mode):
    model = AWQMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=-1, dataset=dataset, scale_estimation=True)


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_gptq(mode):
    model = AWQMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 8, 8])])

    compress_weights(model, mode=mode, ratio=1.0, group_size=2, dataset=dataset, gptq=True)


# TODO(andreyanufr) Waiting for the e2m1 in OV release
@pytest.mark.xfail
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
def test_mixed_precision_e2m1(mode, all_layers, ratio, ref_ids):
    model = SequentialMatmulModel().ov_model
    dataset = Dataset([np.ones([1, 3, 3]), np.arange(9).reshape(3, 3)])
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.E2M1,
        ratio=ratio,
        group_size=1,
        all_layers=all_layers,
        sensitivity_metric=mode,
        dataset=dataset,
    )
    names_e2m1 = {
        op.get_friendly_name() for op in compressed_model.get_ordered_ops() if op.get_element_type() == ov.Type.f4e2m1
    }
    ref_e2m1_nodes = {f"weights_{i}" for i in ref_ids}
    assert ref_e2m1_nodes == names_e2m1

    names_e8m0 = {
        op.get_friendly_name() for op in compressed_model.get_ordered_ops() if op.get_element_type() == ov.Type.f8e8m0
    }
    ref_e8m0_nodes = {f"weights_{i}/scale" for i in ref_ids}
    assert ref_e8m0_nodes == names_e8m0


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM))
def test_np_ov_compression_decompression(mode):
    sz = 60
    w = np.arange(-sz, sz).reshape(2, sz).astype(np.float32) / 9.0
    w = Tensor(w)

    config = WeightCompressionConfig(mode)

    compressed_weighs, scale, zp = do_int_quantization(w, -1, config, invert_scale=True)
    decompressed_weighs = do_int_dequantization(compressed_weighs, scale, zp)

    compressed_weighs = compressed_weighs.data
    decompressed_weighs = decompressed_weighs.data
    zp_shape = zp.shape if zp is not None else None

    compress = OVWeightCompressionAlgoBackend.get_compress_pipeline(config, w.shape, scale.shape, zp_shape)
    compress_decompress = OVWeightCompressionAlgoBackend.get_compress_decompress_pipeline(
        config, w.shape, scale.shape, zp_shape
    )

    params = [w.data, scale.data, zp.data] if zp is not None else [w.data, scale.data]
    compressed_weighs_ov = compress(params)
    decompressed_weighs_ov = compress_decompress(params)

    assert np.allclose(compressed_weighs, compressed_weighs_ov)
    assert np.allclose(decompressed_weighs, decompressed_weighs_ov)


@pytest.mark.parametrize(
    ("mode", "data"),
    (
        (CompressWeightsMode.INT4_SYM, [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]),
        (CompressWeightsMode.INT4_SYM, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            CompressWeightsMode.INT4_SYM,
            [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ),
    ),
)
def test_compressed_weighs_range(mode, data):
    data = np.array(data).astype(np.float32)
    w = Tensor(data)

    config = WeightCompressionConfig(mode=mode)
    compressed_weighs, _, _ = do_int_quantization(w, -1, config)

    assert np.allclose(np.abs(compressed_weighs.data), np.abs(w.data))


@pytest.mark.parametrize(
    ("config", "precompute_scale", "precompute_zero_point", "raises"),
    [
        (WeightCompressionConfig(CompressWeightsMode.INT8_ASYM), False, False, False),
        (WeightCompressionConfig(CompressWeightsMode.INT8_ASYM), True, True, False),
        (WeightCompressionConfig(CompressWeightsMode.INT8_ASYM), True, False, True),
        (WeightCompressionConfig(CompressWeightsMode.INT8_ASYM), False, True, True),
        (WeightCompressionConfig(CompressWeightsMode.INT4_ASYM), False, False, False),
        (WeightCompressionConfig(CompressWeightsMode.INT4_ASYM), True, True, False),
        (WeightCompressionConfig(CompressWeightsMode.INT4_ASYM), True, False, True),
        (WeightCompressionConfig(CompressWeightsMode.INT4_ASYM), False, True, True),
        (WeightCompressionConfig(CompressWeightsMode.INT8_SYM), True, False, False),
        (WeightCompressionConfig(CompressWeightsMode.INT8_SYM), False, False, False),
        (WeightCompressionConfig(CompressWeightsMode.INT4_SYM), True, False, False),
        (WeightCompressionConfig(CompressWeightsMode.INT4_SYM), False, False, False),
    ],
)
def test_int_quantization_with_precomputed_parameters(config, precompute_scale, precompute_zero_point, raises):
    is_asym = config.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]

    precomputed_scale, precomputed_zero_point = None, None
    weight = Tensor(((np.arange(11) - 5) / 10).astype(np.float32)[:, None])
    if precompute_scale:
        precomputed_scale = Tensor(-((np.arange(11) - 5) / 100).astype(np.float32)[:, None])
    if precompute_zero_point:
        precomputed_zero_point = Tensor(np.arange(11).astype(np.int32)[:, None])

    if raises:
        with pytest.raises(ValueError) as exc_info:
            _, scale, zero_point = do_int_quantization(weight, -1, config, precomputed_scale, precomputed_zero_point)
            assert exc_info.value == (
                "If precomputed quantization parameters are provided, both scale and zero point "
                "are required for asymmetric quantization."
            )
        return
    else:
        _, scale, zero_point = do_int_quantization(weight, -1, config, precomputed_scale, precomputed_zero_point)

    if precompute_scale:
        assert np.allclose(scale.data, precomputed_scale.data)
    if is_asym:
        if precompute_zero_point:
            assert np.allclose(zero_point.data, precomputed_zero_point.data)
    else:
        assert zero_point is None


@pytest.mark.parametrize("mode", INT4_NF4_MODES)
def test_call_max_var_criterion_with_dataset_gptq_neg_group_size(mode):
    model = AWQMatmulModel().ov_model
    sz = 8
    dataset = Dataset([np.ones([1, sz, sz])])

    compressed_model = compress_weights(model, mode=mode, ratio=1.0, group_size=-1, dataset=dataset, gptq=True)

    for op in compressed_model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and ("/zero_point" in op_name or "/scale" in op_name):
            assert op.get_shape() == [sz, 1]


@pytest.mark.parametrize("mode", INT4_MODES)
def test_one_dimentional_samples(mode):
    model = AWQMatmulModel().ov_model
    sz = 8
    n_samples = 10
    dataset = Dataset([np.ones([1, i + 1, sz]) for i in range(n_samples)])

    compressed_model = compress_weights(model, mode=mode, ratio=1.0, group_size=-1, dataset=dataset, awq=True)

    for op in compressed_model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and ("/zero_point" in op_name or "/scale" in op_name):
            assert op.get_shape() == [sz, 1]


def test_awq_with_ignored_scope():
    model = AWQMatmulModel().ov_model
    sz = 8
    n_samples = 10
    dataset = Dataset([np.ones([1, i + 1, sz]) for i in range(n_samples)])

    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_ASYM,
        ratio=1.0,
        group_size=-1,
        dataset=dataset,
        awq=True,
        ignored_scope=IgnoredScope(names=["MatMul_6"]),
    )

    act_num = 0
    num_compressed = 8
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.u4:
            act_num += 1
    assert act_num == num_compressed


def get_shape_for_second_input(op_with_weights: ov.Node) -> List[int]:
    return list(op_with_weights.inputs()[1].get_shape())


@pytest.mark.parametrize(
    "params, transpose_b",
    ((None, True), (LoraParams(adapter_rank=4, use_int8_adapters=False), False)),
)
def test_lora_adapters_in_the_graph(params, transpose_b):
    advanced_parameters = CompressionParams() if params is None else CompressionParams(lora_correction_params=params)
    model = LMLinearModel(transpose_b=transpose_b)
    ov_model = model.ov_model
    dataset = Dataset(np.ones(inp.shape) for inp in ov_model.inputs)

    compressed_model = compress_weights(
        ov_model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=8,
        dataset=dataset,
        all_layers=True,
        lora_correction=True,
        advanced_parameters=advanced_parameters,
    )
    input_node = compressed_model.inputs[0].node
    target_inputs = input_node.output(0).get_target_inputs()
    assert len(target_inputs) == 2
    for target_input in target_inputs:
        next_node = target_input.get_node()
        assert next_node.type_info.name == "MatMul"
        shape = get_shape_for_second_input(next_node)
        if shape != model.get_weight_shape(transpose_b):
            assert shape == [advanced_parameters.lora_correction_params.adapter_rank, model.hidden_dim]
            node = get_next_node(next_node)
            assert node.type_info.name == "MatMul"
            assert get_shape_for_second_input(node) == [
                LMLinearModel.OUTPUT_DIM,
                advanced_parameters.lora_correction_params.adapter_rank,
            ]
        else:
            node = get_next_node(next_node)
            assert node.type_info.name == "Add"

    num_u8 = sum(1 for op in compressed_model.get_ordered_ops() if op.get_element_type() == ov.Type.u8)
    ref_u8 = 4 if advanced_parameters.lora_correction_params.use_int8_adapters else 0
    assert ref_u8 == num_u8


@pytest.mark.parametrize(
    "mode, apply_regularization, is_per_channel",
    (
        (CompressWeightsMode.INT4_SYM, True, False),
        (CompressWeightsMode.NF4, True, False),
        (CompressWeightsMode.NF4, True, True),
    ),
)
def test_lora_adapters_reduce_noise(zero_seed, mode, apply_regularization, is_per_channel, mocker, tmp_path):
    mocker.patch("nncf.quantization.algorithms.weight_compression.lora_correction.DEBUG_LOG_DIR", str(tmp_path))

    model = LMLinearModel()
    group_size = -1 if is_per_channel else model.hidden_dim // 2
    model = model.ov_model
    n_iters = 1
    ie = ov.Core()
    input_data = [np.ones(inp.shape) for inp in model.inputs]
    dataset = Dataset(input_data)

    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    fp32_out = infer_request.infer(input_data, share_inputs=True)
    fp32_out = next(iter(fp32_out.values()))
    int4_model = compress_weights(model, mode=mode, ratio=1.0, group_size=group_size, dataset=dataset, all_layers=True)
    compiled_model = ie.compile_model(int4_model, "CPU")
    infer_request = compiled_model.create_infer_request()
    int4_out = infer_request.infer(input_data, share_inputs=True)
    int4_out = next(iter(int4_out.values()))
    noise_before = np.mean(np.abs(fp32_out - int4_out))

    model = LMLinearModel().ov_model

    with nncf_debug():
        int4_model = compress_weights(
            model,
            mode=mode,
            ratio=1.0,
            group_size=group_size,
            dataset=dataset,
            all_layers=True,
            lora_correction=True,
            advanced_parameters=CompressionParams(
                lora_correction_params=LoraParams(
                    apply_regularization=apply_regularization, num_iterations=n_iters, adapter_rank=2
                )
            ),
        )
    compiled_model = ie.compile_model(int4_model, "CPU")
    infer_request = compiled_model.create_infer_request()
    int4_out = infer_request.infer(input_data, share_inputs=True)
    int4_out = next(iter(int4_out.values()))
    noise_after = np.mean(np.abs(fp32_out - int4_out))
    assert noise_after < noise_before

    # expect dumping of quantization noise N times: initial noise + noise after SVD + twice per each iteration
    df = pd.read_csv(tmp_path / "lora" / "noises.csv")
    assert len(df.index) == n_iters * 2 + 2

    # difference between initial and final noise expected to be positive
    df = df.drop(df.columns[[0]], axis=1)
    noise_reduction = df.iloc[0] - df.iloc[-1]
    assert all(noise_reduction > 0)


@pytest.mark.parametrize(
    "activation_dtype, weight_dtype",
    [
        (ov.Type.f32, ov.Type.f32),
        (ov.Type.f32, ov.Type.f16),
        (ov.Type.f32, ov.Type.bf16),
        (ov.Type.f16, ov.Type.f16),
        (ov.Type.bf16, ov.Type.bf16),
    ],
)
def test_compression_with_lora_for_different_dtypes(activation_dtype, weight_dtype):
    model = IdentityMatmul(weights_dtype=weight_dtype, activation_dtype=activation_dtype).ov_model

    data_type = np.float16 if activation_dtype in [ov.Type.f16, ov.Type.bf16] else np.float32
    input_data = [np.ones(inp.shape, dtype=data_type) for inp in model.inputs]
    dataset = Dataset(input_data)

    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=1,
        dataset=dataset,
        all_layers=True,
        lora_correction=True,
        advanced_parameters=CompressionParams(
            lora_correction_params=LoraParams(num_iterations=0, adapter_rank=3, use_int8_adapters=True)
        ),
    )

    name_to_node_map = {op.get_friendly_name(): op for op in compressed_model.get_ops()}
    scale_multiply_node = name_to_node_map["weights/fq_weights_1"]
    check_compressed_matmul_subgraph(scale_multiply_node, activation_dtype, weight_dtype)

    v_adapter_node = name_to_node_map["MatMul_lora_A/fq_weights_1"]
    check_compressed_matmul_subgraph(v_adapter_node, activation_dtype, weight_dtype, True)

    u_adapter_node = name_to_node_map["MatMul_lora_B/fq_weights_1"]
    check_compressed_matmul_subgraph(u_adapter_node, activation_dtype, weight_dtype, True)


def test_compression_with_lora_with_subset_size(mocker):
    subset_size = 2
    dataset_size = 4
    model = LMLinearModel()
    ov_model = model.ov_model
    input_data = [np.ones(inp.shape) for inp in ov_model.inputs] * dataset_size
    dataset = Dataset(input_data)

    from nncf.quantization.algorithms.weight_compression import lora_correction

    get_stats_spy = mocker.spy(lora_correction, "process_stats")

    compress_weights(
        ov_model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=8,
        dataset=dataset,
        all_layers=True,
        lora_correction=True,
        advanced_parameters=CompressionParams(
            lora_correction_params=LoraParams(
                num_iterations=0, adapter_rank=3, use_int8_adapters=False, subset_size=subset_size
            )
        ),
    )

    get_stats_spy.assert_called_once()
    s, X = get_stats_spy.spy_return
    assert X.shape == (model.hidden_dim, subset_size)
    assert s.shape == (model.hidden_dim,)


def test_lora_with_mixed_precision():
    model = AWQMatmulModel().ov_model
    sz = 8
    dataset = Dataset([np.ones([1, sz, sz])])

    compressed_model = compress_weights(
        model, mode=CompressWeightsMode.INT4_ASYM, ratio=0.5, group_size=-1, dataset=dataset, lora_correction=True
    )
    for op in compressed_model.get_ordered_ops():
        op_name = op.get_friendly_name()
        if op.get_type_name() == "Constant" and ("/zero_point" in op_name or "/scale" in op_name):
            assert op.get_shape() == [sz, 1]


@pytest.mark.parametrize("backup_mode", [BackupMode.NONE, BackupMode.INT8_ASYM, BackupMode.INT8_SYM])
def test_data_free_compression_with_backup_mode(backup_mode):
    model = AWQMatmulModel().ov_model
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.NF4,
        ratio=0.7,
        group_size=-1,
        backup_mode=backup_mode,
    )
    act_num = 0
    num_compressed = 3
    if backup_mode == BackupMode.INT8_ASYM:
        backup_ov_mode = ov.Type.u8
    elif backup_mode == BackupMode.INT8_SYM:
        backup_ov_mode = ov.Type.i8
    else:
        backup_ov_mode = ov.Type.f32
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant":
            if op.get_element_type() == ov.Type.nf4:
                act_num += 1
            elif "/scale" in op.get_friendly_name():
                assert op.get_element_type() == ov.Type.f16
            else:
                assert op.get_element_type() == backup_ov_mode
    assert act_num == num_compressed


@pytest.mark.parametrize("backup_mode", [BackupMode.NONE, BackupMode.INT8_ASYM, BackupMode.INT8_SYM])
@pytest.mark.parametrize(
    ("params", "num_compressed"),
    (
        ({"all_layers": True}, 8),
        ({"all_layers": False}, 6),
        ({"sensitivity_metric": SensitivityMetric.HESSIAN_INPUT_ACTIVATION}, 6),
        ({"sensitivity_metric": SensitivityMetric.MEAN_ACTIVATION_VARIANCE}, 6),
        ({"sensitivity_metric": SensitivityMetric.MAX_ACTIVATION_VARIANCE}, 6),
        ({"sensitivity_metric": SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE}, 6),
        ({"scale_estimation": True}, 6),
        ({"lora_correction": True}, 6),
        ({"gptq": True}, 6),
        ({"awq": True}, 6),
    ),
)
def test_data_based_compression_with_backup_mode(backup_mode, params, num_compressed):
    model = AWQMatmulModel().ov_model
    sz = 8
    n_samples = 10
    dataset = Dataset([np.ones([1, i + 1, sz]) for i in range(n_samples)])

    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT4_ASYM,
        ratio=0.8,
        group_size=-1,
        dataset=dataset,
        backup_mode=backup_mode,
        **params,
    )
    act_num = 0
    if backup_mode == BackupMode.INT8_ASYM:
        backup_ov_mode = ov.Type.u8
    elif backup_mode == BackupMode.INT8_SYM:
        backup_ov_mode = ov.Type.i8
    else:
        backup_ov_mode = ov.Type.f32
    for op in compressed_model.get_ops():
        if op.get_type_name() == "Constant":
            if op.get_element_type() == ov.Type.u4:
                act_num += 1
            elif "/scale" in op.get_friendly_name():
                assert op.get_element_type() == ov.Type.f16
            elif "_lora_" in op.get_friendly_name():
                assert op.get_element_type() == ov.Type.u8
            else:
                assert op.get_element_type() == backup_ov_mode
    assert act_num == num_compressed


@pytest.mark.parametrize("n_extra_dims", [0, 1, 2])
def test_data_aware_algo_with_different_activation_dimensions(n_extra_dims):
    model = AWQMatmulModel(n_extra_dims=n_extra_dims).ov_model
    dataset = Dataset([np.ones([1] * n_extra_dims + [8, 8])])
    compress_weights(
        model,
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=-1,
        dataset=dataset,
        awq=True,
        ratio=0.5,
        sensitivity_metric=SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        LMLinearModel.INPUT_SHAPE,
        [3, 5, 16],
        [1, 1, 16],
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale_estimation=True),
        dict(lora_correction=True),
        dict(
            gptq=True,
            scale_estimation=True,
            advanced_parameters=CompressionParams(gptq_params=GPTQParams(subset_size=2)),
        ),
        dict(
            awq=True,
            gptq=True,
            scale_estimation=True,
            advanced_parameters=CompressionParams(gptq_params=GPTQParams(subset_size=2)),
        ),
    ],
)
def test_compression_with_different_algo_combinations(input_shape, kwargs):
    dataset_size = 4
    model = LMLinearModel(input_shape=input_shape).ov_model
    input_data = [np.ones(inp.shape) for inp in model.inputs] * dataset_size
    dataset = Dataset(input_data)

    compress_weights(
        model,
        mode=CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        group_size=8,
        subset_size=2,
        dataset=dataset,
        all_layers=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale_estimation=True),
        dict(lora_correction=True),
        dict(
            gptq=True,
            awq=True,
            scale_estimation=True,
            advanced_parameters=CompressionParams(gptq_params=GPTQParams(subset_size=2)),
        ),
    ],
)
def test_compression_with_transposed_activations(kwargs):
    dataset_size = 4
    model = LMLinearModel(transpose_a=True, transpose_b=False).ov_model
    input_data = [np.ones(inp.shape) for inp in model.inputs] * dataset_size
    dataset = Dataset(input_data)

    with pytest.raises(nncf.UnsupportedModelError):
        compress_weights(
            model,
            mode=CompressWeightsMode.INT4_SYM,
            ratio=1.0,
            group_size=8,
            subset_size=2,
            dataset=dataset,
            all_layers=True,
            **kwargs,
        )
