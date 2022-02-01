import numpy as np

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_layer_name: str, tensor: np.ndarray, is_weights: bool):
        super().__init__(TransformationType.INSERT, target_layer_name)
        self.is_weights = is_weights
        self.tensor = tensor

class ONNXUpdateParameters(TransformationCommand):
    pass
