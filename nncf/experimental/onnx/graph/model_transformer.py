from onnx import ModelProto
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout


class ONNXModelTransformer(ModelTransformer):
    def __init__(self, model: ModelProto):
        super().__init__(model)

    def transform(self, transformation_layout: ONNXTransformationLayout) -> ModelProto:
        ...
