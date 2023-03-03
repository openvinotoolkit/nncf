"""
 Copyright (c) 2023 Intel Corporation
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

import pytest
from pathlib import Path
import numpy as np
import openvino.runtime as ov

from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.openvino_native.statistics.aggregator import OVStatisticsAggregator
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantizationParameters

from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.common import load_json
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import MatMul2DModel
from tests.openvino.native.models import FPModel

REFERENCE_SCALES_DIR = OPENVINO_NATIVE_TEST_ROOT / 'data' / 'reference_scales'


def get_fq_nodes_stats_algo(model):
    nodes = {}
    for op in model.get_ops():
        if op.get_type_name() == 'FakeQuantize':
            input_low = op.input_value(1).get_node().get_data()
            input_high = op.input_value(2).get_node().get_data()
            output_low = op.input_value(3).get_node().get_data()
            output_high = op.input_value(4).get_node().get_data()

            nodes[op.get_friendly_name()] = {
                    'input_low': input_low,
                    'input_high': input_high,
                    'output_low': output_low,
                    'output_high': output_high
            }
    return nodes


def compare_stats(expected, actual):
    assert len(expected) == len(actual)
    for ref_name in expected:
        ref_stats = expected[ref_name]
        ref_input_low, ref_input_high = ref_stats['input_low'], ref_stats['input_high']
        ref_output_low, ref_output_high = ref_stats['output_low'], ref_stats['output_high']

        stats = actual[ref_name]
        input_low, input_high = stats['input_low'], stats['input_high']
        output_low, output_high = stats['output_low'], stats['output_high']

        assert np.allclose(ref_input_low, input_low, atol=1e-6)
        assert np.allclose(ref_input_high, input_high, atol=1e-6)
        assert np.allclose(ref_output_low, output_low, atol=1e-6)
        assert np.allclose(ref_output_high, output_high, atol=1e-6)


# pylint: disable=protected-access
def quantize_model(ov_model, preset):
    dataset = get_dataset_for_test(ov_model)

    min_max_algo = MinMaxQuantization(MinMaxQuantizationParameters(number_samples=1, preset=preset))
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = min_max_algo.get_statistic_points(ov_model)
    statistics_aggregator.register_stastistic_points(statistic_points)
    statistics_aggregator.collect_statistics(ov_model)
    quantized_model = min_max_algo._apply(ov_model, statistics_aggregator.statistic_points)
    return quantized_model


@pytest.mark.parametrize('preset', [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
                         ids=[QuantizationPreset.PERFORMANCE.value, QuantizationPreset.MIXED.value])
@pytest.mark.parametrize('model_creator_func', SYNTHETIC_MODELS.values())
def test_syntetic_models_fq_scales(model_creator_func, preset):
    model = model_creator_func()
    quantized_model = quantize_model(model.ov_model, preset)
    nodes = get_fq_nodes_stats_algo(quantized_model)

    ref_stats_name = model.ref_graph_name.split(".")[0] + f'_{preset.value}.json'
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name
    ref_nodes = load_json(ref_stats_path)

    compare_stats(ref_nodes, nodes)


OMZ_MODELS = [
    'mobilenet-v2-pytorch',
    'resnet-18-pytorch',
    'yolo-v4-tiny-tf',
]


@pytest.mark.parametrize('preset', [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
                         ids=[QuantizationPreset.PERFORMANCE.value, QuantizationPreset.MIXED.value])
@pytest.mark.parametrize('model_name', OMZ_MODELS)
def test_omz_models_fq_scales(model_name, preset, tmp_path):
    _ = download_model(model_name, tmp_path)
    model_path = convert_model(model_name, tmp_path)
    model = ov.Core().read_model(model_path)
    quantized_model = quantize_model(model, preset)
    nodes = get_fq_nodes_stats_algo(quantized_model)

    ref_stats_name = str(Path(model_path).name).rsplit('.', maxsplit=1)[0] + f'_{preset.value}.json'
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name
    ref_nodes = load_json(ref_stats_path)

    compare_stats(ref_nodes, nodes)


REF_NODES_SHAPES = {
    'LinearModel': {'Input/fq_output_0': (), 'MatMul/fq_weights_1': (1, 1, 1, 1)},
    'ConvModel': {'Conv/fq_weights_1': (3, 1, 1, 1), 'Sub/fq_output_0': ()},
    'MatMul2DModel': {'Input/fq_output_0': (), 'MatMul/fq_weights_1': (5, 1)},
}

@pytest.mark.parametrize('model_creator_func, ref_shapes',
                         zip([LinearModel, ConvModel, MatMul2DModel], REF_NODES_SHAPES.values()))
def test_syntetic_models_fq_shapes(model_creator_func, ref_shapes):
    model = model_creator_func()
    quantized_model = quantize_model(model.ov_model, QuantizationPreset.PERFORMANCE)
    nodes = get_fq_nodes_stats_algo(quantized_model)
    for node_name, node in nodes.items():
        assert node['input_low'].shape == ref_shapes[node_name]
        assert node['input_high'].shape == ref_shapes[node_name]
        assert node['output_low'].shape == ref_shapes[node_name]
        assert node['output_high'].shape == ref_shapes[node_name]


@pytest.mark.parametrize('const_dtype', ['FP16', 'FP32'])
@pytest.mark.parametrize('input_dtype', ['FP16', 'FP32'])
def test_fq_precision_orig_fp32model(const_dtype, input_dtype):
    model = FPModel(const_dtype, input_dtype)
    quantized_model = quantize_model(model.ov_model, QuantizationPreset.PERFORMANCE)
    for op in quantized_model.get_ops():
        if op.get_type_name() == 'FakeQuantize':
            inp_node = op.input(0)
            fq_input_node = inp_node.get_source_output().get_node()
            if fq_input_node.get_element_type() == 'Constant':
                assert op.get_element_type() == ov.Type(np.float32 if input_dtype == 'FP32' else np.float16)
        elif op.get_type_name() == 'Convert':
            inp_node = op.input(0)
            fq_input_node = inp_node.get_source_output().get_node()
            if fq_input_node.get_element_type() == 'Constant':
                assert op.get_element_type() == ov.Type(np.float32 if const_dtype == 'FP32' else np.float16)
