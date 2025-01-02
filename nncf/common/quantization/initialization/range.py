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

from typing import Dict, List, Optional, Tuple, Union

from nncf.common.graph.utils import get_reduction_axes
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.config.schemata.defaults import NUM_INIT_SAMPLES
from nncf.experimental.common.tensor_statistics.collectors import AggregationAxes


class RangeInitConfig:
    """
    The `RangeInitConfig` class representing the quantization range initialization
    parameters.
    """

    def __init__(self, init_type: str, num_init_samples: int, init_type_specific_params: Dict = None):
        """
        Initializes the quantization range initialization parameters.

        :param init_type: Type of the initializer - determines which tensor
            statistics will be used to initialize quantization ranges.
        :param num_init_samples: The number of samples from the dataset to consume
            as sample model inputs to compute quantization ranges.
        :param init_type_specific_params: additional parameters specific to the type
            of the initializer
        """
        self.init_type = init_type
        self.num_init_samples = num_init_samples
        self.init_type_specific_params = init_type_specific_params
        if self.init_type_specific_params is None:
            self.init_type_specific_params = {}

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_dict(cls, dct: Dict) -> "RangeInitConfig":
        num_init_samples = dct.get("num_init_samples", NUM_INIT_SAMPLES)
        if num_init_samples < 0:
            raise ValueError("Number of initialization samples must be >= 0")
        return cls(dct.get("type", "mixed_min_max"), num_init_samples, dct.get("params"))


class PerLayerRangeInitConfig(RangeInitConfig):
    """
    The `PerLayerRangeInitConfig` class representing the quantization range
    initialization parameters for layers which are specified using the target
    and ignored scopes and the target group of quantizers.
    """

    def __init__(
        self,
        range_init_config: RangeInitConfig,
        target_scopes: Optional[List[str]],
        ignored_scopes: Optional[List[str]],
        target_quantizer_group: QuantizerGroup = None,
    ):
        """
        Initializes the quantization range initialization parameters.

        :param range_init_config: The quantization range initialization parameters.
        :param target_scopes: A list of model control flow graph node scopes
            to be considered for this operation - functions as a 'denylist'
        :param ignored_scopes: A list of model control flow graph node scopes
            to be ignored for this operation - functions as an 'allowlist'
        :param target_quantizer_group: The target group of quantizers for which
            specified type of range initialization will be applied. It can be
            quantizers group for activations or weights.
        """

        super().__init__(
            range_init_config.init_type, range_init_config.num_init_samples, range_init_config.init_type_specific_params
        )
        if target_scopes is None and ignored_scopes is None:
            raise ValueError(
                "At least one of the (target_scopes, ignored_scopes) should be specified"
                " for a per-layer range init config!"
            )
        self.target_scopes = target_scopes
        self.ignored_scopes = ignored_scopes
        self.target_group = target_quantizer_group

    @classmethod
    def from_dict(cls, dct: Dict) -> "PerLayerRangeInitConfig":
        base_config = RangeInitConfig.from_dict(dct)

        def get_list(dct: Dict, attr_name: str) -> Optional[List[str]]:
            str_or_list = dct.get(attr_name)
            if str_or_list is None:
                return None
            if isinstance(str_or_list, str):
                retval_list = [str_or_list]
            else:
                retval_list = str_or_list
            return retval_list

        target_scopes, ignored_scopes = get_list(dct, "target_scopes"), get_list(dct, "ignored_scopes")

        target_group_str = dct.get("target_quantizer_group")
        target_group = None
        if target_group_str is not None:
            target_group = QuantizerGroup(target_group_str)

        return cls(base_config, target_scopes, ignored_scopes, target_group)


class RangeInitParams:
    """
    The `RangeInitParams` class representing the initialization dataset and the
    quantization range initialization parameters for all model layers.
    """

    def __init__(
        self,
        init_range_data_loader: NNCFDataLoader,
        device: str,
        global_init_config: Optional[RangeInitConfig],
        per_layer_range_init_configs: List[PerLayerRangeInitConfig],
    ):
        """

        :param init_range_data_loader: Provides an iterable over the given dataset.
        :param device: Device to perform initialization. If `device` is `None`
            then the device of the model parameters will be used.
        :param global_init_config: The quantization range initialization parameters
            for a model
        :param per_layer_range_init_configs: The List of the quantization range
            initialization parameters for model layers
        """
        self.init_range_data_loader = init_range_data_loader
        self.device = device
        self.global_init_config = global_init_config
        self.per_layer_range_init_configs = per_layer_range_init_configs


class RangeInitCollectorParams:
    """
    Defines low-level parameters that are used to instantiate statistic collectors.
    """

    def __init__(self, is_weights: bool, scheme: QuantizationScheme, per_channel: bool):
        """
        Initializes Range Initialization Collector Parameters.

        :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.
        :param scheme: Quantization scheme: symmetric or asymmetric.
        :param per_channel: Quantization granularity.
        """
        self._is_weights = is_weights
        self._scheme = scheme
        self._is_per_channel = per_channel

    @property
    def is_weights(self) -> bool:
        """
        Returns boolean that defines tensor type.
        True for Weights, False for Activations.
        """
        return self._is_weights

    @property
    def scheme(self) -> QuantizationScheme:
        """
        Returns quantization scheme: symmetric or asymmetric.
        """
        return self._scheme

    @property
    def is_per_channel(self) -> bool:
        """
        Returns quantization granularity.
        """
        return self._is_per_channel

    def use_per_sample_stats(self, per_sample_stats) -> bool:
        """
        For activations, if per_sample_stats is True, statistics will be collected per-sample.
        For weights statistics are always collected per-batch.

        :param per_sample_stats: Defined by certain collector design.
        :return: A boolean that defines whether to collect statistics per-sample or per-batch.
        """
        return per_sample_stats and (not self._is_weights)

    @property
    def use_abs_max(self) -> bool:
        """Applies abs(max) for symmetric quantization."""
        return self._scheme == QuantizationScheme.SYMMETRIC

    @property
    def use_means_of_mins(self) -> bool:
        return not self._is_weights and not self._is_per_channel and self._scheme == "asymmetric"

    @property
    def use_means_of_maxs(self) -> bool:
        return not self._is_weights and not self._is_per_channel

    def _get_reduction_axes(
        self,
        shape_to_reduce: Union[Tuple[int, ...], List[int]],
        quantization_axes: Union[Tuple[int, ...], List[int]],
        aggregation_axes: Union[Tuple[int, ...], List[int]],
    ):
        """
        Returns axes for a reducer regarding aggregation axes. As aggregator takes axes counting from stacked tensors,
        from these axes only tensor related axes should be used for reducer.

        :param shape_to_reduce: Shape of a reduced tensor.
        :param quantization_axes: Axes of quantization.
        :param aggregation_axes: Axes of aggregator which is applied onto reduced tensor.
        :return: Axes for reducer.
        """
        axes_to_keep = set(el - 1 for el in aggregation_axes if el != 0)
        axes_to_keep.update(quantization_axes)
        return get_reduction_axes(axes_to_keep, shape_to_reduce)

    def _get_aggregation_axes(self, batchwise_statistics: bool) -> Tuple[int, ...]:
        """
        Returns axes for aggregator.

        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch.
        :return Tuple[int]: Aggregation axes.
        """
        return (0, 1) if batchwise_statistics else (0,)

    def get_reduction_aggregation_axes(
        self,
        shape_to_reduce: Union[Tuple[int, ...], List[int]],
        quantization_axes: Union[Tuple[int, ...], List[int]],
        batchwise_statistics: bool,
    ) -> Tuple[ReductionAxes, AggregationAxes]:
        """
        Calculates the reduction axes, aggregation axes for the tensor.

        :param shape_to_reduce: Shape of the tensor.
        :param quantization_axes: Quantization axes if per-channel quantization.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch.
        :return: Reduction axes and aggregation axes.
        """
        aggregation_axes = self._get_aggregation_axes(batchwise_statistics)
        reduction_axes = self._get_reduction_axes(shape_to_reduce, quantization_axes, aggregation_axes)
        return reduction_axes, aggregation_axes
