import numpy
import torch
from nncf import NNCFConfig
from tests.helpers import create_conv, create_compressed_model_and_algo_for_test
import onnxruntime as rt


def get_config_for_export_mode(should_be_onnx_standard: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update({
        "input_info": {
            "sample_size": [1, 1, 20, 20]
        },
        "compression": {
            "algorithm": "quantization",
            "export_to_onnx_standard_ops": should_be_onnx_standard
        },
        "target_device": "CPU"
    })
    return nncf_config


def test_onnx_output_saturation_issue(tmp_path):
    model = torch.nn.Sequential(create_conv(1, 1, 1, 0, 0))

    nncf_config = get_config_for_export_mode(True)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    compressed_model.activation_quantizers['/nncf_model_input_0'].scale.data = torch.tensor(200)
    input_tensor = 1000 * numpy.ones([1, 1, 20, 20])
    torch_input = torch.tensor(input_tensor, dtype=torch.float32)

    with torch.no_grad():
        torch_out = compressed_model.forward(torch_input)

    assert torch.all(torch.eq(torch_out, torch.tensor(200)))

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path, input_names=['input'])

    # ONNXRuntime
    sess = rt.InferenceSession(onnx_checkpoint_path)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: input_tensor.astype(numpy.float32)})[0]

    assert (numpy.allclose(torch_out.numpy(), onnx_out))
