import pytest

import onnx

from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.algorithms.quantization.helper import QuantizerLayerParameters
from nncf.common.quantization.structs import QuantizationMode
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout

from tests.onnx.models import LinearModel

TARGET_LAYERS = ['Non_Existing_Edge', 'Conv1_Y']
SHOULD_RAISE_EXCEPTION = [True, False]


@pytest.mark.parametrize("target_layer, should_raise", zip(TARGET_LAYERS, SHOULD_RAISE_EXCEPTION))
def test_quantizer_insertion(target_layer, should_raise):
    model = LinearModel().onnx_model

    command = ONNXQuantizerInsertionCommand(target_layer,
                                            QuantizerLayerParameters([1.0], [0], QuantizationMode.SYMMETRIC))
    transformation_layout = ONNXTransformationLayout()
    transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)
    if should_raise:
        try:
            _ = model_transformer.transform(transformation_layout)
        except RuntimeError:
            return
    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)

    num_q = 0
    num_dq = 0
    # pylint:disable=no-member
    for node in transformed_model.graph.node:
        op_type = node.op_type
        if op_type == 'QuantizeLinear':
            num_q += 1
        elif op_type == 'DequantizeLinear':
            num_dq += 1
    assert num_q == 1
    assert num_q == num_dq
