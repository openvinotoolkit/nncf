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

from functools import partial
from typing import Deque, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import NNCFTensor
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import AggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MedianAbsoluteDeviationAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import PercentileAggregator
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.statistics import PTMeanTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic


class PTNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for PTNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        return PTNNCFTensor(torch.amin(x.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        return PTNNCFTensor(torch.amax(x.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.abs(x.tensor))

    @classmethod
    def min(cls, *args) -> NNCFTensor:
        stacked = cls.stack(args)
        return cls.reduce_min(stacked, axis=0, keepdims=False)

    @classmethod
    def max(cls, *args) -> NNCFTensor:
        stacked = cls.stack(args)
        return cls.reduce_max(stacked, axis=0, keepdims=False)

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        return PTNNCFTensor(x.tensor.mean(dim=axis, keepdim=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        # See https://github.com/pytorch/pytorch/issues/61582
        if not isinstance(axis, int):
            device = x.tensor.device
            result = torch.tensor(np.median(x.tensor.detach().cpu().numpy(), axis=axis, keepdims=keepdims))
            return PTNNCFTensor(result.type(x.tensor.dtype).to(device))
        return PTNNCFCollectorTensorProcessor.quantile(x, quantile=[0.5], axis=axis, keepdims=keepdims)[0]

    @classmethod
    def masked_mean(
        cls, x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], mask: NNCFTensor, keepdims=False
    ) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        device = x.tensor.device
        masked_x = np.ma.array(x.tensor.detach().cpu().numpy(), mask=mask.tensor.detach().cpu().numpy())
        result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims).astype(masked_x.dtype)
        if isinstance(result, np.ma.MaskedArray):
            result = result.data
        return PTNNCFTensor(torch.tensor(result).to(device=device))

    @classmethod
    def masked_median(
        cls, x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], mask: NNCFTensor, keepdims=False
    ) -> NNCFTensor:
        # Implemented in numy as torch.masked.median is not implemented yet
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        device = x.tensor.device
        masked_x = np.ma.array(x.tensor.detach().cpu().numpy(), mask=mask.tensor.detach().cpu().numpy())
        result = np.ma.median(masked_x, axis=axis, keepdims=keepdims).astype(masked_x.dtype)
        if isinstance(result, np.ma.MaskedArray):
            result = result.data
        return PTNNCFTensor(torch.tensor(result).to(device=device))

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return PTNNCFTensor(torch.mean(x.tensor, axis=0))
        x = torch.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return PTNNCFTensor(torch.mean(t, axis=(0, 2)))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def transpose(x: NNCFTensor, axes: Tuple[int, ...]) -> NNCFTensor:
        return PTNNCFTensor(torch.permute(x.tensor, axes))

    @staticmethod
    def reshape(x: NNCFTensor, shape: Tuple[int, ...]) -> NNCFTensor:
        return PTNNCFTensor(torch.reshape(x.tensor, shape))

    @staticmethod
    def cat(x: List[NNCFTensor], axis: int) -> NNCFTensor:
        x = [t.tensor for t in x]
        return PTNNCFTensor(torch.cat(x, axis))

    @staticmethod
    def logical_or(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.logical_or(input_.tensor, other.tensor))

    @staticmethod
    def less(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(input_.tensor < other.tensor)

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return PTNNCFTensor(torch.stack(x, dim=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        tensor = x.tensor
        if list(tensor.shape) == []:
            tensor = tensor.unsqueeze(0)
        tensor_list = torch.unbind(tensor, dim=axis)
        return [PTNNCFTensor(t) for t in tensor_list]

    @staticmethod
    def squeeze(x: NNCFTensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> NNCFTensor:
        return PTNNCFTensor(torch.squeeze(x.tensor, dim=dim))

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return torch.sum(tensor.tensor).item()

    @staticmethod
    def quantile(
        tensor: NNCFTensor,
        quantile: Union[float, List[float], np.ndarray],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[NNCFTensor]:
        device = tensor.device
        # See https://github.com/pytorch/pytorch/issues/61582
        # https://github.com/pytorch/pytorch/issues/64947
        if len(tensor.tensor) <= 16_000_000 and isinstance(axis, int):
            result = torch.quantile(
                tensor.tensor,
                torch.tensor(quantile, dtype=tensor.tensor.dtype, device=tensor.tensor.device),
                axis,
                keepdims,
            )
        else:
            result = torch.tensor(
                np.quantile(tensor.tensor.detach().cpu().numpy(), q=quantile, axis=axis, keepdims=keepdims)
            )
        result = result.type(tensor.tensor.dtype).to(device)
        return [PTNNCFTensor(x) for x in result]

    @classmethod
    def percentile(
        cls,
        tensor: NNCFTensor,
        percentile: Union[float, List[float], np.ndarray],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        quantile = np.true_divide(percentile, 100)
        return cls.quantile(tensor, quantile=quantile, axis=axis, keepdims=keepdims)

    @staticmethod
    def sub(a: NNCFTensor, b: NNCFTensor) -> NNCFTensor:
        return NNCFTensor(a.tensor - b.tensor)

    @staticmethod
    def zero_elements(x: NNCFTensor) -> NNCFTensor:
        pt_tensor = x.tensor
        eps = torch.finfo(pt_tensor.dtype).eps
        return NNCFTensor(pt_tensor.abs() < eps)


class PTReducerMixIn:
    def _get_processor(self):
        return PTNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class PTNoopReducer(PTReducerMixIn, NoopReducer):
    pass


class PTMinReducer(PTReducerMixIn, MinReducer):
    pass


class PTMaxReducer(PTReducerMixIn, MaxReducer):
    pass


class PTAbsMaxReducer(PTReducerMixIn, AbsMaxReducer):
    pass


class PTMeanReducer(PTReducerMixIn, MeanReducer):
    pass


class PTQuantileReducer(PTReducerMixIn, QuantileReducer):
    pass


class PTAbsQuantileReducer(PTReducerMixIn, AbsQuantileReducer):
    pass


class PTBatchMeanReducer(PTReducerMixIn, BatchMeanReducer):
    pass


class PTMeanPerChanelReducer(PTReducerMixIn, MeanPerChReducer):
    pass


def _reshape_all(targets: Tuple[torch.Tensor, ...], target_shape: Tuple[int, ...]):
    return map(lambda stat: torch.reshape(stat, target_shape), targets)


def _get_wrapped_min_max_tensor_statistic(target_shape: Tuple[int, ...]) -> Type[PTMinMaxTensorStatistic]:
    """
    Returns PTMinMaxTensorStatistic type but all statistics are reshaped to target_shape.

    :param target_shape: Target shape of the tensor statistic
    :return: PTMinMaxTensorStatistic type but all statistics are reshaped to target_shape.
    """

    class WrappedPTMinMaxTensorStatistic(PTMinMaxTensorStatistic):
        def __init__(self, min_values, max_values):
            min_values, max_values = _reshape_all((min_values, max_values), target_shape)
            super().__init__(min_values, max_values)

    return WrappedPTMinMaxTensorStatistic


def _get_wrapped_percentile_tensor_statistic(target_shape: Tuple[int, ...]) -> Type[PTPercentileTensorStatistic]:
    """
    Returns PTPercentileTensorStatistic type but all statistics are reshaped to target_shape.

    :param target_shape: Target shape of the tensor statistic
    :return: PTPercentileTensorStatistic type but all statistics are reshaped to target_shape.
    """

    class WrappedPTPercentileTensorStatistic(PTPercentileTensorStatistic):
        def __init__(self, percentile_vs_values_dict):
            reshaped_percentiles = {}
            for k, v in percentile_vs_values_dict.items():
                reshaped_percentiles[k] = torch.reshape(v, target_shape)
            super().__init__(reshaped_percentiles)

    return WrappedPTPercentileTensorStatistic


def get_min_max_statistic_collector(
    use_abs_max: bool,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
) -> TensorCollector:
    """
    Min max statistic collector builder.

    :param use_abs_max: Whether to use abs max reducer or max reducer.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :return: Min max statistic collector.
    """

    tensor_collector = TensorCollector(_get_wrapped_min_max_tensor_statistic(target_shape=scale_shape))

    aggregator_kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "aggregation_axes": aggregation_axes,
    }
    min_reducer = PTMinReducer(reduction_axes)
    min_aggregator = MinAggregator(**aggregator_kwargs)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = PTAbsMaxReducer if use_abs_max else PTMaxReducer
    max_reducer = max_reducer_cls(reduction_axes)
    max_aggregator = MaxAggregator(**aggregator_kwargs)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)
    return tensor_collector


def get_mixed_min_max_statistic_collector(
    use_abs_max: bool,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    use_means_of_mins: bool,
    use_means_of_maxs: bool,
    num_samples: int = None,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Mixed min max statistic collector builder.

    :param use_abs_max: Whether to use abs max reducer or max reducer.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param use_means_of_mins: Whether to use mean or min aggregator for minimum statistic branch.
    :param use_means_of_maxs: Whether to use mean or max aggregator for maximum statistic branch.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mixed min max statistic collector.
    """
    tensor_collector = TensorCollector(_get_wrapped_min_max_tensor_statistic(target_shape=scale_shape))
    min_reducer = PTMinReducer(reduction_axes)

    kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "aggregation_axes": aggregation_axes,
        "window_size": window_size,
    }
    min_aggregator_cls = MeanAggregator if use_means_of_mins else MinAggregator
    min_aggregator = min_aggregator_cls(**kwargs)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = PTAbsMaxReducer if use_abs_max else PTMaxReducer
    max_reducer = max_reducer_cls(reduction_axes)
    max_aggregator_cls = MeanAggregator if use_means_of_maxs else MaxAggregator
    max_aggregator = max_aggregator_cls(**kwargs)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)

    return tensor_collector


def get_median_mad_statistic_collector(
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Median Absolute Deviation statistic collector builder.

    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Median Absolute Deviation statistic collector.

    """

    class WrappedPTMedianMADTensorStatistic(PTMedianMADTensorStatistic):
        def __init__(self, median_values, mad_values):
            median_values, mad_values = _reshape_all((median_values, mad_values), scale_shape)
            super().__init__(median_values, mad_values)

    return _get_collection_without_reduction(
        MedianAbsoluteDeviationAggregator,
        WrappedPTMedianMADTensorStatistic,
        reduction_axes=reduction_axes,
        aggregation_axes=aggregation_axes,
        num_samples=num_samples,
        window_size=window_size,
    )


def get_percentile_tensor_collector(
    percentiles_to_collect: Tuple[int, ...],
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Percentile statistic collector builder.

    :param percentiles_to_collect: Percetiles to use on aggregation phase.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Percentile statistic collector.
    """
    return _get_collection_without_reduction(
        partial(PercentileAggregator, percentiles_to_collect=percentiles_to_collect),
        _get_wrapped_percentile_tensor_statistic(target_shape=scale_shape),
        reduction_axes=reduction_axes,
        aggregation_axes=aggregation_axes,
        num_samples=num_samples,
        window_size=window_size,
    )


def _get_collection_without_reduction(
    aggregator_cls: AggregatorBase,
    statistic_cls: AggregatorBase,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Helper function to build a tensor collector which is reducing statistics exclusively during aggregation phase.

    :param aggregator_cls: Aggregator class to build the tensor collector.
    :param aggregator_cls: Statistic class to build the tensor collector.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Target statistic collector.
    """
    tensor_collector = TensorCollector(statistic_cls)
    reducer = PTNoopReducer()
    aggregation_axes = list(set(list(aggregation_axes) + [dim + 1 for dim in reduction_axes]))
    aggregator = aggregator_cls(
        PTNNCFCollectorTensorProcessor,
        aggregation_axes=aggregation_axes,
        window_size=window_size,
        num_samples=num_samples,
    )

    tensor_collector.register_statistic_branch(
        PTMedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY, reducer, aggregator
    )
    return tensor_collector


def get_mean_percentile_statistic_collector(
    percentiles_to_collect: Tuple[int, ...],
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Mean percentile statistic collector builder.

    :param percentiles_to_collect: Percetiles to use on reduction phase.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mean percentile statistic collector.
    """
    tensor_collector = TensorCollector(_get_wrapped_percentile_tensor_statistic(target_shape=scale_shape))
    quantiles_to_collect = np.true_divide(percentiles_to_collect, 100)
    reducer = PTQuantileReducer(reduction_axes=reduction_axes, quantile=quantiles_to_collect)
    for output_port_id, p in enumerate(percentiles_to_collect):
        aggregator = MeanAggregator(
            PTNNCFCollectorTensorProcessor,
            aggregation_axes=aggregation_axes,
            num_samples=num_samples,
            window_size=window_size,
        )
        tensor_collector.register_statistic_branch(
            (PTPercentileTensorStatistic.PERCENTILE_VS_VALUE_DICT, p), reducer, aggregator, output_port_id
        )
    return tensor_collector


def get_mean_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None
) -> TensorCollector:
    """
    Mean statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mean statistic collector.
    """
    if channel_axis == 0:
        reducer = PTBatchMeanReducer()
    else:
        reducer = PTMeanPerChanelReducer(channel_axis=channel_axis)
    noop_reducer = PTNoopReducer()

    kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(PTMeanTensorStatistic)
    collector.register_statistic_branch(PTMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(PTMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


PT_REDUCERS_MAP = {
    StatisticsType.MIN: PTMinReducer,
    StatisticsType.MAX: PTMaxReducer,
    StatisticsType.ABS_MAX: PTAbsMaxReducer,
    StatisticsType.MEAN: PTMeanReducer,
    StatisticsType.QUANTILE: PTQuantileReducer,
    StatisticsType.ABS_QUANTILE: PTAbsQuantileReducer,
}
