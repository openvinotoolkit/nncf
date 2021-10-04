import numpy as np

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs, enumerate_model_node_outputs
import onnx
import onnxruntime as rt

def collect_tensor_statistics(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    scale = (max_val - min_val) / 256
    zero_point = 0
    return scale, zero_point


def calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader, num_iters):
    onnx_model = nncf_network.onnx_model
    model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=[outputs[0], model_output])
    onnx.save(model_with_intermediate_outputs, '/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')

    sess = rt.InferenceSession('/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')
    input_name = sess.get_inputs()[0].name
    avg_scale, zero_point = 0, 0
    for i, (input_, target) in enumerate(data_loader):
        input_tensor = input_.cpu().detach().numpy()
        img_size = [1, 3, 224, 224]

        output_tensor = sess.run([], {input_name: input_tensor.astype(np.float32)})
        scale, zero_point = collect_tensor_statistics(output_tensor[0])
        avg_scale += scale
        if i == num_iters:
            break
    return avg_scale / num_iters, zero_point