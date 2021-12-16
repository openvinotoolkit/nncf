import torchvision
import torch
import tempfile
import onnx


def test_post_training_quantization_onnx(mocker, sota_data_dir):
    from examples.experimental.post_training_quantization_onnx.classification.post_training_quantization import run
    mocker.patch(
        'nncf.experimental.onnx.quantization.algorithm.calculate_statistics_for_activation_quantizer',
        return_value=(1, 0))
    model = torchvision.models.resnet50(pretrained=True)
    number_of_q_in_quantized_model = 141
    with tempfile.NamedTemporaryFile() as temporary_model:
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, temporary_model.name, opset_version=13)
        with tempfile.NamedTemporaryFile() as temporary_quantized_model:
            num_init_samples = 1
            input_shape = [1, 3, 224, 224]
            run(temporary_model.name, temporary_quantized_model.name, sota_data_dir,
                num_init_samples, input_shape)
            onnx.checker.check_model(temporary_quantized_model.name)

            onnx_model_proto = onnx.load(temporary_quantized_model.name)
            num_q = 0
            num_dq = 0
            num_model_nodes = 0
            # pylint:disable=no-member
            for node in onnx_model_proto.graph.node:
                op_type = node.op_type
                if op_type == 'QuantizeLinear':
                    num_q += 1
                elif op_type == 'DequantizeLinear':
                    num_dq += 1
                elif op_type in ['Conv', 'Constant']:
                    num_model_nodes += 1
            assert num_q == number_of_q_in_quantized_model == num_dq
