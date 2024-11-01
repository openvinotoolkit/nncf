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

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Tuple

from nncf.tensor import Tensor
from nncf.tensor import functions as fns


class TensorStatistic:
    """Base class that stores statistic data"""

    TENSOR_STATISTIC_OUTPUT_KEY = "tensor_statistic_output"

    def get_data(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}

    def load_data(self, data: Dict[str, Any]):
        for key in self.keys():
            setattr(self, key, data.get(key))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TensorStatistic:
        args = {key: config[key] for key in cls.keys()}  # noqa: SIM118
        return cls(**args)

    @classmethod
    def keys(cls) -> Tuple[str]:
        return ()


@dataclass
class MinMaxTensorStatistic(TensorStatistic):
    MIN_STAT: ClassVar[str] = "min_values"
    MAX_STAT: ClassVar[str] = "max_values"

    min_values: Tensor
    max_values: Tensor

    @classmethod
    def keys(cls):
        return (cls.MIN_STAT, cls.MAX_STAT)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MinMaxTensorStatistic):
            return fns.allclose(self.min_values, other.min_values) and fns.allclose(self.max_values, other.max_values)
        return False


@dataclass
class AbsMaxTensorStatistic(TensorStatistic):
    ABS_MAX_STAT: ClassVar[str] = "abs_max"

    abs_max: Tensor

    @classmethod
    def keys(cls):
        return (cls.ABS_MAX_STAT,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, AbsMaxTensorStatistic):
            return fns.allclose(self.abs_max, other.abs_max)
        return False


@dataclass
class MeanTensorStatistic(TensorStatistic):
    MEAN_STAT: ClassVar[str] = "mean_values"
    SHAPE_STAT: ClassVar[str] = "shape"

    mean_values: Tensor
    shape: Tuple[int, ...]

    @classmethod
    def keys(cls):
        return (cls.MEAN_STAT, cls.SHAPE_STAT)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MeanTensorStatistic):
            return self.shape == other.shape and fns.allclose(self.mean_values, other.mean_values)
        return False


@dataclass
class MedianMADTensorStatistic(TensorStatistic):
    MEDIAN_VALUES_STAT: ClassVar[str] = "median_values"
    MAD_VALUES_STAT: ClassVar[str] = "mad_values"

    median_values: Tensor
    mad_values: Tensor

    @classmethod
    def keys(cls):
        return (cls.MEDIAN_VALUES_STAT, cls.MAD_VALUES_STAT)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MedianMADTensorStatistic):
            return fns.allclose(self.median_values, other.median_values) and fns.allclose(
                self.mad_values, other.mad_values
            )
        return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TensorStatistic:
        return cls(
            median_values=config[cls.TENSOR_STATISTIC_OUTPUT_KEY][cls.MEDIAN_VALUES_STAT],
            mad_values=config[cls.TENSOR_STATISTIC_OUTPUT_KEY][cls.MAD_VALUES_STAT],
        )


@dataclass
class PercentileTensorStatistic(TensorStatistic):
    PERCENTILE_VS_VALUE_DICT: ClassVar[str] = "percentile_vs_values_dict"

    percentile_vs_values_dict: Dict[str, Tensor]

    @classmethod
    def keys(cls):
        return (cls.PERCENTILE_VS_VALUE_DICT,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, PercentileTensorStatistic):
            if Counter(self.percentile_vs_values_dict.keys()) != Counter(other.percentile_vs_values_dict.keys()):
                return False
            for pct in self.percentile_vs_values_dict:
                if not fns.allclose(self.percentile_vs_values_dict[pct], other.percentile_vs_values_dict[pct]):
                    return False
            return True
        return False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TensorStatistic:
        if cls.TENSOR_STATISTIC_OUTPUT_KEY in config:
            percentile_vs_values_dict = config[cls.TENSOR_STATISTIC_OUTPUT_KEY]
        else:
            percentile_vs_values_dict = {}
            for (_, percentile), value in config.items():
                percentile_vs_values_dict[percentile] = value
        return cls(percentile_vs_values_dict=percentile_vs_values_dict)


@dataclass
class RawTensorStatistic(TensorStatistic):
    VALUES_STATS: ClassVar[str] = "values"

    values: Tensor

    @classmethod
    def keys(cls):
        return (cls.VALUES_STATS,)

    def __eq__(self, other: RawTensorStatistic) -> bool:
        if isinstance(other, RawTensorStatistic):
            return fns.allclose(self.values, other.values)
        return False


@dataclass
class HessianTensorStatistic(TensorStatistic):
    HESSIAN_INPUT_ACTIVATION_STATS: ClassVar[str] = "hessian"

    hessian: Tensor

    @classmethod
    def keys(cls):
        return (cls.HESSIAN_INPUT_ACTIVATION_STATS,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, HessianTensorStatistic):
            return fns.allclose(self.hessian, other.hessian)
        return False


@dataclass
class MeanVarianceTensorStatistic(TensorStatistic):
    MEAN_VARIANCE_STAT: ClassVar[str] = "mean_variance"

    mean_variance: Tensor

    @classmethod
    def keys(cls):
        return (cls.MEAN_VARIANCE_STAT,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MeanVarianceTensorStatistic):
            return fns.allclose(self.mean_variance, other.mean_variance)
        return False


@dataclass
class MaxVarianceTensorStatistic(TensorStatistic):
    MAX_VARIANCE_STAT: ClassVar[str] = "max_variance"

    max_variance: Tensor

    @classmethod
    def keys(cls):
        return (cls.MAX_VARIANCE_STAT,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MaxVarianceTensorStatistic):
            return fns.allclose(self.max_variance, other.max_variance)
        return False


@dataclass
class MeanMagnitudeTensorStatistic(TensorStatistic):
    MEAN_MAGNITUDE_STAT: ClassVar[str] = "mean_magnitude"

    mean_magnitude: Tensor

    @classmethod
    def keys(cls):
        return (cls.MEAN_MAGNITUDE_STAT,)

    def __eq__(self, other: TensorStatistic):
        if isinstance(other, MeanMagnitudeTensorStatistic):
            return fns.allclose(self.mean_magnitude, other.mean_magnitude)
        return False


@dataclass
class WCTensorStatistic(TensorStatistic):
    MEAN_STAT = "mean_values"
    SHAPE_STAT = "shape_values"

    mean_values: List[Tensor]
    shape_values: List[Tuple[int, ...]]

    @classmethod
    def keys(cls):
        return (cls.MEAN_STAT, cls.SHAPE_STAT)

    def __eq__(self, other: Any) -> bool:
        shapes_equal = all(self.shapes[i] == other.shapes[i] for i in range(len(self.mean_values)))
        if not shapes_equal:
            return False
        mean_values_equal = all(
            self.tensor_eq(self.mean_values[i], other.mean_values[i]) for i in range(len(self.mean_values))
        )
        return mean_values_equal

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> TensorStatistic:
        mean_values, shape_values = None, None
        if cls.MEAN_STAT in config and config[cls.MEAN_STAT] is not None:
            mean_values = [fns.squeeze(it) for it in config[cls.MEAN_STAT]]
        if cls.SHAPE_STAT in config and config[cls.SHAPE_STAT] is not None:
            shape_values = [tuple(it) for it in config[cls.SHAPE_STAT]]
        return cls(mean_values=mean_values, shape_values=shape_values)
