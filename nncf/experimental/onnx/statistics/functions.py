from typing import List

import numpy as np

from nncf.experimental.post_training.statistics.statistics_collector import CalculateTensorValueFunc
from nncf.experimental.post_training.statistics.statistics_collector import BatchAggregatorFunc
from nncf.experimental.post_training.statistics.statistics_collector import StatisticsCalculationFunc


class ONNXTensorMinFunc(CalculateTensorValueFunc):
    @staticmethod
    def __call__(tensor: np.ndarray, axis: int):
        return np.min(tensor, axis=axis)


class ONNXTensorMaxFunc(CalculateTensorValueFunc):
    @staticmethod
    def __call__(tensor: np.ndarray, axis: int):
        return np.max(tensor, axis=axis)


class ONNXBatchMaxFunc(BatchAggregatorFunc):
    @staticmethod
    def __call__(tensor: np.ndarray):
        return np.max(tensor, axis=0)


class ONNXBatchMinFunc(BatchAggregatorFunc):
    @staticmethod
    def __call__(tensor: np.ndarray):
        return np.min(tensor, axis=0)


class ONNXBatchMeanFunc(BatchAggregatorFunc):
    @staticmethod
    def __call__(tensor: np.ndarray):
        return np.mean(tensor, axis=0)


class ONNXStatisticsMeanFunc(StatisticsCalculationFunc):
    @staticmethod
    def __call__(tensors: List[np.ndarray]) -> float:
        return np.mean(tensors)


class ONNXStatisticsABSMAXFunc(StatisticsCalculationFunc):
    @staticmethod
    def __call__(tensors: List[np.ndarray]) -> float:
        # TODO: need to test
        return np.where(np.max(np.abs(tensors)), tensors)
