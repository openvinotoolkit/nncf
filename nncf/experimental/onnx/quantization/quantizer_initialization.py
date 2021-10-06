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


def calculate_statistics_for_activation_quantizer(onnx_model: onnx.ModelProto, outputs, data_loader, num_iters,
                                                  mode='min_max'):
    model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=[outputs[0], model_output])
    # TODO: add temporary location and also deleting this model
    onnx.save(model_with_intermediate_outputs,
              '/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')

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
        return statistics_collector.global_max, statistics_collector.global_min
    elif mode == 'mean_min_max':
        return statistics_collector.max_avg, statistics_collector.min_avg
    raise RuntimeError('')


def calculate_statistics_for_weight_quantizer(weight_tensor: np.ndarray, num_bits: int):
    filter_max, filter_min = [], []
    for filter in weight_tensor:
        filter_max.append(np.max(filter))
        filter_min.append(np.min(filter))
    return calculate_scale_level(np.array(filter_max), np.array(filter_min), num_bits)


def calculate_scale_level(max_val: float, min_val: float, num_bits: int):
    return (max_val - min_val) / 2 ** num_bits
