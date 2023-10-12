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

from typing import Dict, List, Optional, TypeVar

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class WeightCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to nf4 depending on mode, ratio and group size.
    """

    def __init__(
        self,
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
    ):
        """
        :param mode: Defines a mode for weight compression.
            INT8 stands for 8-bit integer quantization of all weights.
            NF4 stands for a mixed-precision weights quantization to NF4 data type. The first and last layers
            are always compressed to a backup precision which is 8-bit integer by default. All others are quantized
            whether to NF4 or to a backup precision depending on criteria and the given ratio.
        :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        """
        super().__init__()
        self._mode = mode
        self._group_size = group_size
        self._ratio = ratio
        self._backend_entity = None
        self._algorithm_key = f"CW_{hash(self)}"

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        self._backend_entity.validate_params(self._mode)
        nodes_to_compress = self._get_nodes_to_compress(graph)
        transformed_model = self._backend_entity.do_compression(
            model, nodes_to_compress, self._mode, self._ratio, self._group_size
        )
        return transformed_model

    def _get_nodes_to_compress(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        weighted_metatypes = self._backend_entity.weighted_metatypes
        ordered_nodes_to_compress = []
        for node in nncf_graph.topological_sort():
            is_node_with_weights = self._backend_entity.is_node_with_weights(node)
            if node.metatype in weighted_metatypes and is_node_with_weights:
                ordered_nodes_to_compress.append(node)
        return ordered_nodes_to_compress

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
