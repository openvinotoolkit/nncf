from abc import ABC
from typing import Dict

import torch


class TensorStatistic(ABC):
    pass


class MinMaxTensorStatistic(TensorStatistic):
    def __init__(self, min_values: torch.Tensor, max_values: torch.Tensor):
        self.min_values = min_values
        self.max_values = max_values

    @classmethod
    def from_stat(cls, statistic: TensorStatistic):
        if isinstance(statistic, MinMaxTensorStatistic):
            return cls(statistic.min_values, statistic.max_values)
        if isinstance(statistic, MedianMADTensorStatistic):
            return cls(statistic.median_values - 3 * statistic.mad_values,
                       statistic.median_values + 3 * statistic.mad_values)
        if isinstance(statistic, PercentileTensorStatistic):
            if len(statistic.percentile_vs_values_dict.keys()) < 2:
                raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
            min_pct = min(statistic.percentile_vs_values_dict.keys())
            max_pct = max(statistic.percentile_vs_values_dict.keys())
            return cls(statistic.percentile_vs_values_dict[min_pct],
                       statistic.percentile_vs_values_dict[max_pct])
        raise ValueError("Unknown statistic to generate min-max stat from!")

class MedianMADTensorStatistic(TensorStatistic):
    def __init__(self, median_values: torch.Tensor, mad_values: torch.Tensor):
        self.median_values = median_values
        self.mad_values = mad_values


class PercentileTensorStatistic(TensorStatistic):
    def __init__(self, percentile_vs_values_dict: Dict[float, torch.Tensor]):
        self.percentile_vs_values_dict = percentile_vs_values_dict
