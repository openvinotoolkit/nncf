"""
 Copyright (c) 2022 Intel Corporation
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

from typing import List, Tuple

import numpy as np

from nncf import Dataset

from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm  import PostTrainingQuantizationParameters

from tests.openvino.native.models import LinearModel

INPUT_SHAPE = [3, 10, 10]

DATASET_SAMPLES = [(np.zeros(INPUT_SHAPE, dtype=np.float32), 0),
                   (np.zeros(INPUT_SHAPE, dtype=np.float32), 1),
                   (np.zeros(INPUT_SHAPE, dtype=np.float32), 2)]

DATASET_SAMPLES[0][0][0, 0, 0] = 128  # max
DATASET_SAMPLES[0][0][0, 0, 1] = -128  # min

DATASET_SAMPLES[1][0][0, 0, 0] = 1  # max
DATASET_SAMPLES[1][0][0, 0, 1] = -10  # min

DATASET_SAMPLES[2][0][0, 0, 0] = 0.1  # max
DATASET_SAMPLES[2][0][0, 0, 1] = -1  # min


class TestParameters:
    def __init__(self, number_samples, activation_float_range, weight_float_range):
        self.number_samples = number_samples
        self.activation_float_range = activation_float_range
        self.weight_float_range = weight_float_range


def get_dataset_for_test(samples: List[Tuple[np.ndarray, int]], input_name: str):

    def transform_fn(data_item):
        inputs, targets = data_item
        return {input_name: [inputs], "targets": targets}
    return Dataset(samples, transform_fn)


@pytest.mark.parametrize('range_type, test_parameters',
                         ((RangeType.MEAN_MINMAX, (TestParameters(1, 128, 1))),
                          (RangeType.MEAN_MINMAX, (TestParameters(2, 69, 1))),
                          (RangeType.MEAN_MINMAX, (TestParameters(3, 46.333, 1))),
                          (RangeType.MINMAX, (TestParameters(1, 128, 1))),
                          (RangeType.MINMAX, (TestParameters(2, 128, 1))),
                          (RangeType.MINMAX, (TestParameters(3, 128, 1)))
                          )
                         )
def test_statistics_aggregator(range_type, test_parameters):
    model = LinearModel().ov_model

    dataset = get_dataset_for_test(DATASET_SAMPLES, "Input")

    # compression_builder = CompressionBuilder()
    # quantization = MinMaxQuantization(MinMaxQuantizationParameters(
    #     number_samples=test_parameters.number_samples,
    #     range_type=range_type
    # ))
    # compression_builder.add_algorithm(quantization)
    # quantized_model = compression_builder.apply(model, dataset)

    quantization_parameters = PostTrainingQuantizationParameters(
        number_samples=test_parameters.number_samples,
        range_type=range_type
    )

    quantization_algorithm = PostTrainingQuantization(quantization_parameters)
    quantized_model = quantization_algorithm.apply(model, dataset=dataset)

    num_q = 0
    for node in quantized_model.get_ops():
        if node.get_type_name() == 'FakeQuantize':
            num_q += 1

            # activation_scale = test_parameters.activation_float_range / ((2 ** 8 - 1) / 2)
            # assert np.allclose(onnx_graph.get_initializers_value(node.input[1]), np.array(activation_scale))
    assert num_q == 2






# def create_quantized_model(model, statistics_aggregator, target_points):
#     tensor_collector = OVMinMaxStatisticCollector(use_abs_max=True, reduction_shape=None, num_samples=3)
#     statistic_points = StatisticPointsContainer()
#     transformation_layout = TransformationLayout()

#     for target_point in target_points:
#         stat_point = StatisticPoint(target_point=target_point,
#                                     tensor_collector=tensor_collector,
#                                     algorithm=PostTrainingAlgorithms.MinMaxQuantization)
#         statistic_points.add_statistic_point(stat_point)
#     statistics_aggregator.register_stastistic_points(statistic_points)
#     model_transformer = OVModelTransformer(model)
#     statistics_aggregator.collect_statistics(model_transformer)

#     for algo_stat_points in statistic_points.values():
#         for statistic_point in algo_stat_points:
#             for tensor_collector in statistic_point.algorithm_to_tensor_collectors[PostTrainingAlgorithms.MinMaxQuantization]:
#                 parameters = calculate_activation_quantizer_parameters(tensor_collector.get_statistics(), num_bits=8)
#                 transformation_commands = OVQuantizerInsertionCommand(statistic_point.target_point, parameters)
#                 transformation_layout.register(transformation_commands)

#     quantized_model = model_transformer.transform(transformation_layout)
#     return quantized_model