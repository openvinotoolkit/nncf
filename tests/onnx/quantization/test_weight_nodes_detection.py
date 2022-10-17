import pytest
from collections import Counter

from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization, \
    MinMaxQuantizationParameters
from tests.onnx.models import WEIGHT_DETECTION_MODELS

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import infer_model


@pytest.mark.parametrize('model_to_test', WEIGHT_DETECTION_MODELS.values())
def test_weight_nodes_detection(model_to_test):
    model_to_test = model_to_test()
    onnx_model = model_to_test.onnx_model
    quantization_algo = ONNXMinMaxQuantization(MinMaxQuantizationParameters(number_samples=1))
    quantizer_setup = quantization_algo._get_quantizer_setup(onnx_model)

    quantized_weight_nodes = []
    for quantization_point in quantizer_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            quantized_weight_nodes.append(quantization_point.insertion_point.target_node_name)

    assert Counter(quantized_weight_nodes) == Counter(model_to_test.weight_nodes)


@pytest.mark.parametrize('model_to_test', WEIGHT_DETECTION_MODELS.values())
def test_matmul_quantization(model_to_test):
    model_to_test = model_to_test()
    onnx_model = model_to_test.onnx_model
    quantized_model = min_max_quantize_model(model_to_test.input_shape[0], onnx_model)
    infer_model(model_to_test.input_shape[0], quantized_model)
