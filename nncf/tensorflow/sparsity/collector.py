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

from typing import List

import tensorflow as tf

from nncf.common.sparsity.collector import BaseSparseModelStatisticsCollector
from nncf.common.sparsity.collector import WeightDescription
from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMaskWithWeightsBackup


def _get_standardized_weight_shape(shape):
    return [0 if x is None else x for x in shape]


class TFSparseModelStatisticsCollector(BaseSparseModelStatisticsCollector):
    """
    Collects statistics for the sparse tf.keras.Model.
    """

    def __init__(self, model: tf.keras.Model, operation_names: List[str]):
        """
        Initializes statistics collector of the sparse tf.keras.Model.

        :param model: Sparse model.
        :param operation_names: Names of operations.
        """
        self._model = model
        self._operation_names = operation_names
        self._excluded_names = []
        self._sw_name_to_num_nonzero_map = {}

    def _collect_weights_descriptions(self) -> List[WeightDescription]:
        weights_descriptions = []
        excluded_names = []

        # Collect description for sparse weights i.e. weights for which
        # sparsity algorithm was applied.
        for wrapped_layer, weight_attr, op in get_nncf_operations(self._model, self._operation_names):
            weight = wrapped_layer.layer_weights[weight_attr]
            operation_weights = wrapped_layer.get_operation_weights(op.name)
            binary_mask = op.get_binary_mask(operation_weights)
            sparse_weight = apply_mask(weight, binary_mask)

            weights_descriptions.append(
                WeightDescription(
                    weight.name,
                    _get_standardized_weight_shape(weight.shape.as_list()),
                    tf.math.count_nonzero(sparse_weight).numpy().item(),
                    is_sparse=True,
                )
            )

            # Exclude this name because it has been processed.
            excluded_names.append(weight.name)

            # Exclude these names because they were added to the model
            # by the sparsity algorithm.
            excluded_names.extend([w.name for w in operation_weights.values()])
            if isinstance(op, BinaryMaskWithWeightsBackup):
                excluded_names.append(op.bkup_var.name)

        # Collect descriptions for rest weights.
        unique_weights = {id(w): w for w in self._model.weights if w.name not in excluded_names}.values()
        for weight in unique_weights:
            weights_descriptions.append(
                WeightDescription(
                    weight.name,
                    _get_standardized_weight_shape(weight.shape.as_list()),
                    tf.math.count_nonzero(weight).numpy().item(),
                    is_sparse=False,
                )
            )

            return weights_descriptions
