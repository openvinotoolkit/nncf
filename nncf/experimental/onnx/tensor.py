import numpy as np

from nncf.common.tensor import NNCFTensor


class ONNXNNCFTensor(NNCFTensor):
    """
    """

    def __init__(self, tensor: np.ndarray):
        super().__init__(tensor)

    @property
    def device(self):
        return 'CPU'
