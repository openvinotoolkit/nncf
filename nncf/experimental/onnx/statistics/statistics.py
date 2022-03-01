import numpy as np

from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic


class ONNXMinMaxTensorStatistic(MinMaxTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: np.ndarray, tensor2: np.ndarray, rtol=1e-6) -> bool:
        return bool(np.allclose(tensor1, tensor2, rtol=rtol))
