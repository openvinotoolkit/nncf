import numpy as np

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs, enumerate_model_node_outputs
import onnx
import onnxruntime as rt


class StatisticsCollector:
    def __init__(self):
        self.global_min = 1e10
        self.global_max = -1e10
        self.min_sum = 0
        self.max_sum = 0
        self.min_avg = 0
        self.max_avg = 0
        self.counter = 0

    def update(self, tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        self.min_sum += min_val
        self.max_sum += max_val
        self.counter += 1
        self.min_avg = self.min_sum / self.counter
        self.max_avg = self.max_sum / self.counter
        self.global_min = min(self.global_min, min_val)
        self.global_max = max(self.global_max, max_val)


def collect_tensor_statistics(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    scale = (max_val - min_val) / 256
    zero_point = 0
    return scale, zero_point, min_val, max_val


def min_max_statistics(tensor):
    global_min, global_max = 1e10, 1e10
    min_val = np.min(tensor)
    max_val = np.max(tensor)

    return global_min, global_max


def calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader, num_iters, mode):
    onnx_model = nncf_network.onnx_model
    model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=[outputs[0], model_output])
    onnx.save(model_with_intermediate_outputs, '/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')

    sess = rt.InferenceSession('/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')
    input_name = sess.get_inputs()[0].name
    statistics_collector = StatisticsCollector()

    for i, (input_, target) in enumerate(data_loader):
        input_tensor = input_.cpu().detach().numpy()
        output_tensor = sess.run([], {input_name: input_tensor.astype(np.float32)})
        statistics_collector.update(output_tensor[0])
        if i == num_iters:
            break
    if mode == 'min_max':
        scale = (statistics_collector.global_max - statistics_collector.global_min) / 256
        return scale, 0
    elif mode == 'mean_min_max':
        scale = (statistics_collector.max_avg - statistics_collector.min_avg) / 256
        return scale, 0
    raise RuntimeError('')