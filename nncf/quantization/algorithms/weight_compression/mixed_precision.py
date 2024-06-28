# Copyright (c) 2024 Intel Corporation
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
from typing import Dict, List, Optional, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.utils.registry import Registry
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")
MIXED_PRECISION_CRITERIA = Registry("mixed_precision_criteria")
THE_LOWEST_SENSITIVITY = 0


class MixedPrecisionCriterion:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or uniform int4/non-uniform fp4)
    for weights based on some criteria.
    """

    def __init__(
        self,
        model: TModel,
        graph: NNCFGraph,
        backend_entity: WeightCompressionAlgoBackend,
        weight_params: List[WeightCompressionParameters],
        primary_config: WeightCompressionConfig,
        ratio: float,
        activations: Optional[Dict[str, List[Tensor]]] = None,
    ):
        """
        :param model: The model.
        :param graph: The model graph associated with the model.
        :param backend_entity: The instance of the WeightCompressionAlgoBackend.
        :param weight_params: Weight compression parameters which determines how and what weight should be compressed.
        :param primary_config: Configuration on how to compress (quantize) weights to primary precision.
        :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param activations: The input activations of the nodes to be quantized.
        """
        self._model = model
        self._graph = graph
        self._backend_entity = backend_entity
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

    def _calc_weight_sensitivity(self, weight_param: WeightCompressionParameters) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, self._model, self._graph
        )
        backup_config = weight_param.compression_config
        reduction_axes = weight_param.reduction_axes
        int_error = get_integer_quantization_error(weight, reduction_axes, backup_config)
        eps = fns.finfo(weight).eps
        return 1 / (int_error + eps)

    def _calc_score_per_node(self, weight_param: WeightCompressionParameters) -> float:
        weight_score = self._calc_weight_sensitivity(weight_param)
        return weight_score

    def _calc_sensitivity(self) -> List[float]:
        scores = []
        for weight_param in track(self._weight_params, description="Mixed-Precision assignment"):
            scores.append(self._calc_score_per_node(weight_param))
        return scores


class DataBasedCriterion(DataFreeCriterion):
    """
    Data-based mixed precision criterion that takes into account outliers in the input activations.
    Expecting activations of the following shape: [seq_length, hidden_dim]
    """

    @staticmethod
    @abstractmethod
    def _calc_activation_sensitivity(activations: List[Tensor]) -> float:
        pass

    def _calc_score_per_node(self, weight_param: WeightCompressionParameters):
        """
        NOTE: Data-based criteria for assigning 4-bit/8-bit precisions are valid for Matmul operations only.
        However, in some cases it can be beneficial to quantize Gather layers to 4-bit.
        Since there's no data-aware estimation of sensitivity in these layers, they receive the lowest sensitivity.
        It allows assigning Gather operation 4-bit in the first place.
        """
        if weight_param.node_with_weight.metatype in self._backend_entity.embedding_metatypes:
            return THE_LOWEST_SENSITIVITY
        weight_score = self._calc_weight_sensitivity(weight_param)
        activation_score = self._calc_activation_sensitivity(self._activations[weight_param.node_with_weight.node_name])
        return weight_score * activation_score


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
class HAWQCriterion(DataBasedCriterion):
    """
    Calculates the average Hessian trace of weights with respect to the layer-wise quantization error
    multiplied by L2 norm of 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: List[Tensor]) -> float:
        htrace = 0
        nsamples = len(activations)
        for inp in activations:
            # NOTE: average trace?? divide by number of diagonal elements
            htrace += fns.sum(fns.multiply(inp, inp)).item()
            # normalize by sequence_length - the same for all activations
            # normalize by hidden dimension
            htrace /= inp.size
        htrace *= 2 / nsamples
        return htrace

    def _calc_weight_sensitivity(self, weight_param: WeightCompressionParameters) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, self._model, self._graph
        )
        backup_config = weight_param.compression_config
        reduction_axes = weight_param.reduction_axes

        orig_shape = weight.shape

        if weight.dtype != TensorDataType.float32:
            weight = weight.astype(TensorDataType.float32)

        compressed_weights, scale, zero_point = do_integer_quantization(weight, reduction_axes, backup_config)
        decompressed_weight = do_dequantization(compressed_weights, scale, zero_point)
        decompressed_weight = decompressed_weight.reshape(orig_shape)
        return fns.linalg.norm(decompressed_weight - weight, ord="fro").item()


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_VARIANCE)
class MeanVarianceCriterion(DataBasedCriterion):
    """
    The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: List[Tensor]) -> float:
        return fns.mean(fns.stack([fns.mean(fns.var(inp, axis=0)) for inp in activations])).item()


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
class MaxVarianceCriterion(DataBasedCriterion):
    """
    The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: List[Tensor]) -> float:
        return fns.mean(fns.stack([fns.max(fns.var(inp, axis=0)) for inp in activations])).item()


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE)
class MeanMaxCriterion(DataBasedCriterion):
    """
    The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    @staticmethod
    def _calc_activation_sensitivity(activations: List[Tensor]) -> float:
        return fns.mean(fns.stack([fns.mean(fns.max(fns.abs(inp), axis=0)) for inp in activations])).item()
