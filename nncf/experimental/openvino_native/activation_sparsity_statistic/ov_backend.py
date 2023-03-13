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

import nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes as ovm
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.backend import BackendType
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
