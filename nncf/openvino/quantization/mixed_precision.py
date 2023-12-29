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

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
from numpy import linalg

from nncf.common.logging.track_progress import track
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.node_utils import get_const_value
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.compression_info import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.compression_info import WeightNodeParams
from nncf.quantization.algorithms.weight_compression.quantize import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.quantize import get_integer_quantization_error

MIXED_PRECISION_CRITERIA = Registry("mixed_precision_criteria")

THE_LOWEST_SENSITIVITY = 0


class MixedPrecisionCriterion:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or non-uniform nf4) for weights based on some criteria.
    """

    def __init__(
        self,
        weight_params: List[WeightNodeParams],
        primary_config: WeightCompressionConfig,
        ratio: float,
        activations: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        :param weight_params: Information about weights that are used for calculating ratio between primary and backup
            precisions. The quantization scheme is added to this info.
        :param primary_config: Information on how to compress (quantize) weights to primary precision.
        :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param activations: The input activations of the nodes to be quantized.
        """
        self._weight_params = weight_params
        self._activations = activations
        self._primary_config = primary_config
        self._ratio = ratio

    @abstractmethod
    def _calc_sensitivity(self) -> List[float]:
        """
        Calculates sensitivity of each layer according to a criterion.

        :return: List of values per node to be quantized.
        """

    def assign_mixed_precision(self) -> None:
        """
        Assigns quantization precision based on computed layers' sensitivities, ratio of parameters.
        """
        scores = self._calc_sensitivity()
        num_all_weights = sum(wp.num_weights for wp in self._weight_params)

        indexes_of_layers_in_ascending_order_of_scores = [
            i[0] for i in sorted(enumerate(scores), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_4bit = 0
        for index in indexes_of_layers_in_ascending_order_of_scores:
            weight_param = self._weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_all_weights
            if current_ratio >= self._ratio:
                break
            weight_param.compression_config = self._primary_config
            num_weights_in_4bit += weight_param.num_weights


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.WEIGHT_QUANTIZATION_ERROR)
class DataFreeCriterion(MixedPrecisionCriterion):
    """
    A baseline mixed precision criterion that is based on quantization noise of weights only.
    """

    @staticmethod
    def _calc_weight_sensitivity(weight_param: WeightNodeParams) -> float:
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axis = weight_param.reduction_axis
        int_error = get_integer_quantization_error(weight, reduction_axis, backup_config)
        eps = np.finfo(weight.dtype).eps
        return 1 / (int_error + eps)

    def _calc_score_per_node(self, weight_param: WeightNodeParams) -> float:
        weight_score = self._calc_weight_sensitivity(weight_param)
        return weight_score

    def _calc_sensitivity(self) -> List[float]:
        scores = []
        for weight_param in track(self._weight_params, description="Searching for Mixed-Precision Configuration"):
            scores.append(self._calc_score_per_node(weight_param))
        return scores


class DataBasedCriterion(DataFreeCriterion):
    """
    Data-based mixed precision criterion that takes into account outliers in the input activations.
    Expecting activations of the following shape: [seq_length, hidden_dim]
    """

    @staticmethod
    @abstractmethod
    def _calc_activation_sensitivity(activations: np.ndarray):
        pass

    def _calc_score_per_node(self, weight_param: WeightNodeParams):
        """
        NOTE: Data-based criteria for assigning 4-bit/8-bit precisions are valid for Matmul operations only.
        However, in some cases it can be beneficial to quantize Gather layers to 4-bit.
        Since there's no data-based estimation of sensitivity in these layers, they receive the lowest sensitivity.
        It allows assigning Gather operation 4-bit in the first place.
        """
        if weight_param.metatype == OVEmbeddingMetatype:
            return THE_LOWEST_SENSITIVITY
        weight_score = self._calc_weight_sensitivity(weight_param)
        activation_score = self._calc_activation_sensitivity(self._activations[weight_param.node_name])
        return weight_score * activation_score


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
class HAWQCriterion(DataBasedCriterion):
    """
    Calculates the average Hessian trace of weights with respect to the layer-wise quantization error
    multiplied by L2 norm of 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: np.ndarray):
        htrace = 0
        nsamples = len(activations)
        for inp in activations:
            # NOTE: average trace?? divide by number of diagonal elements
            htrace += np.sum(np.multiply(inp, inp))
            # normalize by sequence_length - the same for all activations
            # normalize by hidden dimension
            htrace /= inp.size
        htrace *= 2 / nsamples
        return htrace

    @staticmethod
    def _calc_weight_sensitivity(weight_param: WeightNodeParams):
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axis = weight_param.reduction_axis

        orig_shape = weight.shape
        compressed_weights, scale, zero_point = do_integer_quantization(weight, reduction_axis, backup_config)
        decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
        decompressed_weight = (compressed_weights - zero_point) * scale
        decompressed_weight = decompressed_weight.reshape(orig_shape)
        return linalg.norm(decompressed_weight - weight, ord="fro")


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_VARIANCE)
class MeanVarianceCriterion(DataBasedCriterion):
    """
    The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: np.ndarray):
        return float(np.mean([np.mean(np.var(inp, axis=0)) for inp in activations]))


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
class MaxVarianceCriterion(DataBasedCriterion):
    """
    The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: np.ndarray):
        return float(np.mean([np.max(np.var(inp, axis=0)) for inp in activations]))


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE)
class MeanMaxCriterion(DataBasedCriterion):
    """
    The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: np.ndarray):
        return float(np.mean([np.mean(np.max(np.abs(inp), axis=0)) for inp in activations]))
