from typing import Dict

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import tempfile

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase

from nncf.experimental.post_training.statistics.statistics_aggregator import StatisticsAggregator

from nncf.experimental.onnx.sampler import create_onnx_sampler
from nncf.experimental.onnx.engine import ONNXEngine


class ONNXStatisticsAggregator(StatisticsAggregator):
    def __init__(self, engine: ONNXEngine):
        super().__init__(engine)

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        layers_to_collect_statistics = list(self.layers_statistics.keys())
        model_output = list(enumerate_model_node_outputs(model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])
        max_number_samples = 0
        for k, v in self.layers_statistics.items():
            max_number_samples = max(max_number_samples, v._num_samples)

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            sampler = create_onnx_sampler(self.engine)
            for i, sample in enumerate(sampler):
                if i == max_number_samples:
                    break
                _input, target = sample
                output = self.engine.infer(_input)
                self._agregate_statistics(output, self.layers_statistics)

    def _agregate_statistics(self, output, layers_statistics: Dict[str, TensorStatisticCollectorBase]):
        for k, v in layers_statistics.items():
            tensor = output[k]
            v.register_input(tensor)
