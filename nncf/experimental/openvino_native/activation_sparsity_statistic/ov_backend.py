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

from typing import List
from typing import Optional

import openvino.runtime as ov

import nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes as ovm
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import ActivationSparsityStatistic
from nncf.experimental.openvino_native.activation_sparsity_statistic.backend import ALGO_BACKENDS
from nncf.experimental.openvino_native.activation_sparsity_statistic.backend import \
    ActivationSparsityStatisticAlgoBackend
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.statistics.collectors import OVPercentageOfZerosStatisticCollector

ACTIVATION_SPARSITY_STATISTIC = "activation_sparsity_statistic"
DEFAULT_TARGET_NODE_TYPES = [
    ovm.OVConvolutionBackpropDataMetatype,
    ovm.OVConvolutionMetatype,
    ovm.OVDepthwiseConvolutionMetatype,
    ovm.OVGroupConvolutionBackpropDataMetatype,
    ovm.OVGroupConvolutionMetatype,
    ovm.OVMatMulMetatype,
]


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVActivationSparsityStatisticAlgoBackend(ActivationSparsityStatisticAlgoBackend):
    @staticmethod
    def percentage_of_zeros_statistic_collector(
        num_samples: Optional[int] = None,
    ) -> OVPercentageOfZerosStatisticCollector:
        return OVPercentageOfZerosStatisticCollector(num_samples)

    @staticmethod
    def target_point(target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name, port_id)

    @staticmethod
    def default_target_node_types() -> List[str]:
        node_types = []
        for mt in DEFAULT_TARGET_NODE_TYPES:
            node_types.extend(mt.op_names)

        return node_types

    @staticmethod
    def ignored_input_node_types() -> List[str]:
        return ovm.OVParameterMetatype.op_names + ["Constant"]

    @staticmethod
    def write_statistic_to_model(
        model: ov.Model, statistic_points: StatisticPointsContainer, threshold: float
    ) -> ov.Model:
        """
        Write statistics values to rt_info of the target model.
        In the serialized model file, the statistics will be saved as:
            <rt_info>
                <item_0>
                    <node_name value="/nncf_module/maxpool/MaxPool" />
                    <port_id value="0" />
                    <statistic value="0.192276" />
                </item_0>
                <item_1>
                    <node_name value="/nncf_module/layer1/layer1.0/conv1/Conv/WithoutBiases" />
                    <port_id value="0" />
                    <statistic value="0.11276" />
                </item_1>
            </rt_info>

        :param model: Target model.
        :param statistic_points: StatisticPointsContainer instance with the statistic points.
        :param threshold: Threshold of minimum value of statistic that will be save to the model.

        :return: Modified model.
        """
        count_items = 0
        for node_op in model.get_ordered_ops():
            node_name = node_op.get_friendly_name()
            node_static_points = statistic_points.get(node_name, [])

            for node_static_point in node_static_points:
                assert len(node_static_point.algorithm_to_tensor_collectors[ActivationSparsityStatistic]) == 1
                tensor_collector = node_static_point.algorithm_to_tensor_collectors[ActivationSparsityStatistic][0]

                statistic = tensor_collector.get_statistics()
                percentage_of_zeros = statistic.percentage_of_zeros
                if percentage_of_zeros >= threshold:
                    port_id = node_static_point.target_point.port_id
                    model.set_rt_info(
                        percentage_of_zeros, [ACTIVATION_SPARSITY_STATISTIC, f"item_{count_items}", "statistic"]
                    )
                    model.set_rt_info(port_id, [ACTIVATION_SPARSITY_STATISTIC, f"item_{count_items}", "port_id"])
                    model.set_rt_info(node_name, [ACTIVATION_SPARSITY_STATISTIC, f"item_{count_items}", "node_name"])
                    count_items += 1

        return model
