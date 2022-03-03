import numpy as np

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.onnx.algorithms.quantization.helper import QuantizerLayerParameters


class ONNXInsertionCommand(TransformationCommand):
    def __init__(self, target_layer_name: str):
        super().__init__(TransformationType.INSERT, target_layer_name)


class ONNXQuantizerInsertionCommand(ONNXInsertionCommand):
    def __init__(self, target_layer_name: str, quantizer_parameters: QuantizerLayerParameters):
        super().__init__(target_layer_name)
        self.quantizer_parameters = quantizer_parameters
