from typing import Dict

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import tempfile

from nncf.experimental.post_training.statistics.statistics_collector import StatisticsCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector

from nncf.experimental.onnx.sampler import create_onnx_sampler
from nncf.experimental.onnx.engine import ONNXEngine


class ONNXStatisticsCollector(StatisticsCollector):
    def __init__(self, engine: ONNXEngine, number_samples: int):
        super().__init__(engine, number_samples)

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        layers_to_collect_statistics = [list(layer.keys())[0] for layer in self.layers_statistics]
        model_output = list(enumerate_model_node_outputs(model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            sampler = create_onnx_sampler(self.engine)
            for i, sample in enumerate(sampler):
                if i == self.number_samples:
                    break
                _input, target = sample
                output = self.engine.infer(_input)
                self._agregate_statistics(output, self.layers_statistics)

    def _agregate_statistics(self, output, layers_statistics: Dict[str, ONNXMinMaxStatisticCollector]):
        for layer_statistic in layers_statistics:
            for k, v in layer_statistic.items():
                tensor = output[k]
                v.register_input(tensor)
