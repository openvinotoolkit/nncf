import pytest
import torchvision
import torch
import tempfile
import onnx


def test_post_training_quantization_onnx(sota_data_dir):
    from examples.experimental.post_training_quantization_onnx.post_training_quantization import run
    from nncf.experimental.onnx.quantization.quantizer_initialization import \
        calculate_statistics_for_activation_quantizer
    calculate_statistics_for_activation_quantizer = lambda arg1, arg2, arg3, arg4, arg5: (1, 0)

    model = torchvision.models.resnet50(pretrained=True)
    temporary_model = tempfile.NamedTemporaryFile()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, temporary_model.name, opset_version=10)
    temporary_quantized_model = tempfile.NamedTemporaryFile()
    num_init_samples = 1
    input_shape = [1, 3, 224, 224]
    run(temporary_model.name, temporary_quantized_model.name, sota_data_dir, num_init_samples, input_shape)
    onnx.checker.check_model(temporary_quantized_model.name)

    temporary_model.close()
    temporary_quantized_model.close()
