from typing import List

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import tempfile

from nncf.experimental.post_training.statistics.statistics_collector import StatisticsCollector
from nncf.experimental.post_training.statistics.statistics_collector import MinMaxLayerStatistic

from nncf.experimental.onnx.sampler import create_onnx_sampler


class ONNXStatisticsCollector(StatisticsCollector):
    def __init__(self, compressed_model, engine):
        super().__init__(compressed_model, engine)

    def collect_statistics(self, layers_statistics: List[MinMaxLayerStatistic], num_iters: int) -> List[
        MinMaxLayerStatistic]:
        layers_to_collect_statistics = [layer_statistic.layer_name for layer_statistic in layers_statistics]
        onnx_model = self.compressed_model.original_model
        model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            sampler = create_onnx_sampler(self.engine)
            for i, sample in enumerate(sampler):
                if i == num_iters:
                    break
                _input, target = sample
                output = self.engine.infer(_input)
                self._agregate_statistics(output, layers_statistics)
                if self.is_calculate_metric:
                    self._calculate_metric(target)

        return layers_statistics

    def _agregate_statistics(self, output, layers_statistics: List[MinMaxLayerStatistic]):
        for layer_statistic in layers_statistics:
            tensor = output[layer_statistic.layer_name]
            layer_statistic.add_tensor_statistic(tensor)

    def _calculate_metric(self, target):
        pass
