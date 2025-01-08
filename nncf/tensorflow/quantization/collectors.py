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

from typing import List, Tuple

import tensorflow as tf

from nncf.common.quantization.collectors import QuantizationStatisticsCollector
from nncf.common.quantization.collectors import QuantizerDescription
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.quantization.utils import collect_fake_quantize_layers


class TFQuantizationStatisticsCollector(QuantizationStatisticsCollector):
    """
    Implementation of the quantization statistics collector for the TensorFlow backend.
    """

    def __init__(self, model: tf.keras.Model, operation_names: List[str]):
        """
        Initializes a collector of the quantization statistics.

        :param model: An instance of the `tf.keras.Model` class.
        :param operation_names: List of names of quantizer operations. It includes
            names of weight and non-weight quantizers.
        """
        self._model = model
        self._operation_names = operation_names

    def _collect_quantizers_descriptions(self) -> List[QuantizerDescription]:
        """
        Collects descriptions of the quantizers.

        :return: Descriptions of the quantizers.
        """
        quantizers_descriptions = []

        for wrapped_layer, _, op in get_nncf_operations(self._model, self._operation_names):
            is_symmetric = op.mode == QuantizationMode.SYMMETRIC

            is_signed = True
            if is_symmetric:
                operation_weights = wrapped_layer.get_operation_weights(op.name)
                is_signed = op.signed(operation_weights)

            quantizers_descriptions.append(
                QuantizerDescription(op.num_bits, op.per_channel, is_signed, is_symmetric, True, op.enabled)
            )

        for fq_layer in collect_fake_quantize_layers(self._model):
            is_symmetric = fq_layer.mode == QuantizationMode.SYMMETRIC

            quantizers_descriptions.append(
                QuantizerDescription(
                    fq_layer.num_bits, fq_layer.per_channel, fq_layer.signed, is_symmetric, False, fq_layer.enabled
                )
            )

        return quantizers_descriptions

    def _get_potential_quantizers_num(self) -> Tuple[int, int]:
        """
        Returns a potential number of quantizers for weights and activations.

        :return: A tuple (wq_potential_num, aq_potential_num) where
            - `wq_potential_num` is a potential number of quantizers for weights.
            - `aq_potential_num` is a potential number of quantizers for activations.
        """
        return None, None
