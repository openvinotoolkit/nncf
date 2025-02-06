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

from pathlib import Path

import openvino as ov
import pytest
import torch

from nncf.common.quantization.structs import QuantizationPreset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.parameters import QuantizationMode
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from tests.cross_fw.shared.comparator import compare_stats
from tests.cross_fw.shared.json import load_json
from tests.openvino.native.common import convert_torch_model
from tests.openvino.native.common import get_actual_reference_for_current_openvino
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import MatMul2DModel
from tests.openvino.native.models import UnifiedScalesModel
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.models import get_torch_model_info

REFERENCE_SCALES_DIR = Path("reference_scales")


def get_fq_nodes_stats_algo(model):
    nodes = {}
    for op in model.get_ops():
        if op.get_type_name() == "FakeQuantize":
            input_low = op.input_value(1).get_node().data
            input_high = op.input_value(2).get_node().data
            output_low = op.input_value(3).get_node().data
            output_high = op.input_value(4).get_node().data

            nodes[op.get_friendly_name()] = {
                "input_low": input_low,
                "input_high": input_high,
                "output_low": output_low,
                "output_high": output_high,
            }
        elif op.get_type_name() == "FakeConvert":
            scale = op.input_value(1).get_node().data
            shift = op.input_value(2).get_node().data

            nodes[op.get_friendly_name()] = {
                "scale": scale,
                "shift": shift,
            }
    return nodes


def quantize_model(ov_model, q_params):
    dataset = get_dataset_for_test(ov_model)
    graph = GraphConverter.create_nncf_graph(ov_model)

    min_max_algo = MinMaxQuantization(subset_size=1, **q_params)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = min_max_algo.get_statistic_points(ov_model, graph)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(ov_model, graph)
    quantized_model = min_max_algo.apply(ov_model, graph, statistics_aggregator.statistic_points)
    return quantized_model


@pytest.fixture(params=[True, False], ids=["inplace", "out_of_place"], name="inplace_statistics")
def fixture_inplace_statistics(request):
    return request.param


@pytest.mark.parametrize(
    "preset",
    [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
    ids=[QuantizationPreset.PERFORMANCE.value, QuantizationPreset.MIXED.value],
)
@pytest.mark.parametrize("model_creator_func", SYNTHETIC_MODELS.values())
def test_synthetic_models_fq_scales(model_creator_func, preset, inplace_statistics):
    model = model_creator_func()
    quantized_model = quantize_model(model.ov_model, {"preset": preset, "inplace_statistics": inplace_statistics})
    nodes = get_fq_nodes_stats_algo(quantized_model)

    ref_stats_name = model.ref_graph_name.split(".")[0] + f"_{preset.value}.json"
    ref_stats_path = get_actual_reference_for_current_openvino(REFERENCE_SCALES_DIR / ref_stats_name)

    # Uncomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    compare_stats(ref_nodes, nodes)


@pytest.mark.parametrize(
    "mode",
    [QuantizationMode.FP8_E4M3, QuantizationMode.FP8_E5M2],
    ids=[QuantizationMode.FP8_E4M3.value, QuantizationMode.FP8_E5M2.value],
)
@pytest.mark.parametrize("model_creator_func", [UnifiedScalesModel])
def test_synthetic_models_fc_scales(model_creator_func, mode):
    model = model_creator_func()
    quantized_model = quantize_model(model.ov_model, {"mode": mode})
    real_nodes = [op for op in quantized_model.get_ops() if op.get_type_name() == "FakeConvert"]

    ref_stats_name = model.ref_graph_name.split(".")[0] + f"_{mode.value}.json"
    ref_stats_path = get_actual_reference_for_current_openvino(REFERENCE_SCALES_DIR / ref_stats_name)
    ref_nodes = load_json(ref_stats_path)

    assert len(ref_nodes) == len(real_nodes), "The number of the real FakeConvert nodes is not correct"
    stat_nodes = get_fq_nodes_stats_algo(quantized_model)

    # Uncomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    compare_stats(ref_nodes, stat_nodes)


@pytest.mark.parametrize(
    "overflow_fix",
    [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
    ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value],
)
def test_overflow_fix_scales(overflow_fix):
    model = WeightsModel()
    quantized_model = quantize_model(model.ov_model, {"overflow_fix": overflow_fix})
    nodes = get_fq_nodes_stats_algo(quantized_model)

    ref_stats_name = model.ref_graph_name.split(".")[0] + f"_overflow_fix_{overflow_fix.value}.json"
    ref_stats_path = get_actual_reference_for_current_openvino(REFERENCE_SCALES_DIR / ref_stats_name)

    # Uncomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    compare_stats(ref_nodes, nodes)


@pytest.mark.parametrize(
    "preset",
    [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
    ids=[QuantizationPreset.PERFORMANCE.value, QuantizationPreset.MIXED.value],
)
@pytest.mark.parametrize("model_name", ("mobilenet-v2", "resnet-18", "ssd-vgg-300"))
def test_real_models_fq_scales(model_name, preset, inplace_statistics, tmp_path):
    torch.manual_seed(0)  # To use the same initialized model
    model_cls, input_shape = get_torch_model_info(model_name)
    ov_model = convert_torch_model(model_cls(), input_shape, tmp_path)

    quantized_model = quantize_model(ov_model, {"preset": preset, "inplace_statistics": inplace_statistics})
    nodes = get_fq_nodes_stats_algo(quantized_model)
    ref_stats_name = model_name + f"_{preset.value}.json"
    ref_stats_path = get_actual_reference_for_current_openvino(REFERENCE_SCALES_DIR / ref_stats_name)

    # Uncomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    compare_stats(ref_nodes, nodes)


REF_NODES_SHAPES = {
    "LinearModel": {"Input/fq_output_0": (), "MatMul/fq_weights_1": (1, 5)},
    "ConvModel": {"Conv/fq_weights_1": (3, 1, 1, 1), "Sub/fq_output_0": ()},
    "MatMul2DModel": {"Input/fq_output_0": (), "MatMul/fq_weights_1": (1, 2)},
}


@pytest.mark.parametrize(
    "model_creator_func, ref_shapes", zip([LinearModel, ConvModel, MatMul2DModel], REF_NODES_SHAPES.values())
)
def test_synthetic_models_fq_shapes(model_creator_func, ref_shapes, inplace_statistics):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": inplace_statistics}
    )
    nodes = get_fq_nodes_stats_algo(quantized_model)
    for node_name, node in nodes.items():
        assert node["input_low"].shape == ref_shapes[node_name]
        assert node["input_high"].shape == ref_shapes[node_name]
        assert node["output_low"].shape == ref_shapes[node_name]
        assert node["output_high"].shape == ref_shapes[node_name]


@pytest.mark.parametrize("const_dtype", [ov.Type.f16, ov.Type.f32, ov.Type.bf16])
@pytest.mark.parametrize("input_dtype", [ov.Type.f16, ov.Type.f32, ov.Type.bf16])
def test_fq_precision_orig_fp32model(const_dtype, input_dtype, inplace_statistics):
    model = FPModel(const_dtype, input_dtype)
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": inplace_statistics}
    )
    for op in quantized_model.get_ops():
        if op.get_type_name() == "FakeQuantize":
            inp_node = op.input(0)
            fq_input_node = inp_node.get_source_output().get_node()
            if fq_input_node.get_type_name() == "Constant":
                assert op.get_element_type() == const_dtype
        elif op.get_type_name() == "Convert":
            inp_node = op.input(0)
            fq_input_node = inp_node.get_source_output().get_node()
            if fq_input_node.get_type_name() == "Constant":
                assert op.get_element_type() == input_dtype
