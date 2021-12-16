import math
from typing import Union
import numpy as np
# pylint: disable=import-error
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import onnxruntime as rt
import tempfile


class StatisticsCollector:
    def __init__(self):
        self.global_min = math.inf
        self.global_max = -math.inf
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
                                                  mode='mean_min_max'):
    model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=[outputs[0], model_output])
    with tempfile.NamedTemporaryFile() as temporary_model:
        onnx.save(model_with_intermediate_outputs, temporary_model.name)
        sess = rt.InferenceSession(temporary_model.name, providers=['OpenVINOExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        statistics_collector = StatisticsCollector()
        for i, (input_, *other) in enumerate(data_loader):
            if i == num_iters:
                break
            input_tensor = input_.cpu().detach().numpy()
            output_tensor = sess.run([], {input_name: input_tensor.astype(np.float32)})
            statistics_collector.update(output_tensor[0])

    if mode == 'min_max':
        return statistics_collector.global_max, statistics_collector.global_min
    if mode == 'mean_min_max':
        return statistics_collector.max_avg, statistics_collector.min_avg
    raise RuntimeError('Invalid statistics collection mode')


def calculate_statistics_for_weight_quantizer(weight_tensor: np.ndarray, num_bits: int, per_channel: bool = True):
    # Symmetric quantization to range [-128; 127]
    if per_channel:
        scales, zero_points = [], []
        for single_filter in weight_tensor:
            input_high = np.max(single_filter)
            input_low = np.min(single_filter)
            scales.append(calculate_scale_level(input_high, input_low, num_bits, symmetric=True))
            zero_points.append(0)
        return np.array(scales), np.array(zero_points)
    return calculate_scale_level(np.max(weight_tensor), np.min(weight_tensor), num_bits, symmetric=False)


def calculate_scale_level(max_val: Union[float, np.ndarray],
                          min_val: Union[float, np.ndarray],
                          num_bits: int,
                          symmetric: bool):
    # Always full range
    if symmetric:
        input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
        return input_abs_max / ((2 ** num_bits - 1) / 2)
    return (max_val - min_val) / 2 ** num_bits
