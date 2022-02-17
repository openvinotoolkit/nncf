from typing import List
from onnx import NodeProto

import numpy as np

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_layer_name: str):
        super().__init__(TransformationType.INSERT, target_layer_name)


class ONNXQuantizerInsertionCommand(ONNXInsertionCommand):
    def __init__(self, target_layer_name: str, parameters):
        super().__init__(target_layer_name)
        self.parameters = parameters


class ONNXUpdateBias(ONNXInsertionCommand):
    def __init__(self, target_layer_name: str, bias_tensor: np.ndarray):
        super().__init__(target_layer_name)
        self.bias_tensor = bias_tensor
