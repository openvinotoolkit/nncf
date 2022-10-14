import pytest

from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization, \
    MinMaxQuantizationParameters
from tests.onnx.models import WEIGHT_DETECTION_MODELS

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import infer_model
@pytest.mark.parametrize('model_to_test', WEIGHT_DETECTION_MODELS.values())
def test_weight_nodes_detection(model_to_test):
    model_to_test = model_to_test()
    onnx_model = model_to_test.onnx_model
    # quantization_algo = ONNXMinMaxQuantization(MinMaxQuantizationParameters(number_samples=1))
    quantized_model = min_max_quantize_model(model_to_test.input_shape, onnx_model)
    infer_model(model_to_test.input_shape, quantized_model)
    # quantizer_setup = quantization_algo._get_quantizer_setup(onnx_model)

