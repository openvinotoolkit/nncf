from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import tempfile
import numpy as np
import math


class Statistics:
    def __init__(self, _min, _max):
        self.min = min(math.inf, _min)
        self.max = max(-math.inf, _max)


class ONNXStatisticsCollector:
    def __init__(self, compressed_model, engine):
        self.compressed_model = compressed_model
        self.engine = engine
        self.is_calculate_metric = False
        self.statistics = {}

    def collect_statistics(self, layers_to_collect_statistics, num_iters):
        onnx_model = self.compressed_model.original_model
        model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            for i in range(num_iters):
                output, target = self.engine.infer_model(i)
                # output should be topologically sorted
                self._agregate_statistics(output, layers_to_collect_statistics)
                if self.is_calculate_metric:
                    self._calculate_metric(target)

    def _agregate_statistics(self, output, layers_to_collect_statistics):
        for layer_name, function in layers_to_collect_statistics.items():
            if function == 'min_max':
                if self.statistics.get(layer_name, None) is None:
                    self.statistics[layer_name] = Statistics(np.min(output[layer_name]), np.max(output[layer_name]))
                else:
                    self.statistics[layer_name].min = min(self.statistics[layer_name].min, np.min(output[layer_name]))
                    self.statistics[layer_name].max = max(self.statistics[layer_name].max, np.max(output[layer_name]))

    def _calculate_metric(self, target):
        pass
