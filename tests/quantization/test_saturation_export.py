import numpy
import torch
from nncf import NNCFConfig
from tests.helpers import create_conv, create_compressed_model_and_algo_for_test
from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer, \
    QuantizerExportMode
import onnxruntime as rt
from nncf.quantization.layers import SymmetricQuantizer

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
    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)
    quantizer.set_export_mode(QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS)
    model = torch.nn.Sequential(quantizer)
    nncf_config = get_config_for_export_mode(True)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model,nncf_config)
    compressed_model.get_nncf_wrapped_model()

    compressed_model.get_nncf_wrapped_model()[0].scale.data = torch.tensor(200)
    input = 1000 * numpy.ones([1, 1, 20, 20])
    torch_input = torch.tensor(input, dtype=torch.float32)

    with torch.no_grad():
        torch_out = compressed_model.forward(torch_input)

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path)

    # ONNXRuntime
    sess = rt.InferenceSession(onnx_checkpoint_path, None)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: input.astype(numpy.float32)})[0]

    assert (numpy.allclose(torch_out.numpy(), onnx_out))

    q_config = QuantizerConfig(
        bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=True,
        per_channel=False,
        is_weights=True,
        is_saturation_fix=True)

    quantizer = SymmetricQuantizer(q_config)
    quantizer.set_export_mode(QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS)
    model = torch.nn.Sequential(quantizer)
    nncf_config = get_config_for_export_mode(True)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model,nncf_config)
    compressed_model.get_nncf_wrapped_model()

    compressed_model.get_nncf_wrapped_model()[0].scale.data = torch.tensor(200)
    input = 1000 * numpy.ones([1, 1, 20, 20])
    torch_input = torch.tensor(input, dtype=torch.float32)

    with torch.no_grad():
        torch_out = compressed_model.forward(torch_input)

    onnx_checkpoint_path = str(tmp_path / 'model.onnx')
    compression_ctrl.export_model(onnx_checkpoint_path)

    # ONNXRuntime
    sess = rt.InferenceSession(onnx_checkpoint_path, None)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: input.astype(numpy.float32)})[0]

    assert (numpy.allclose(torch_out.numpy(), onnx_out))
