from typing import List

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import tempfile
import numpy as np

from nncf.experimental.post_training.initialization.statistics_collector import StatisticsCollector
from nncf.experimental.post_training.initialization.statistics_collector import LayerStatistic
from nncf.experimental.post_training.initialization.statistics_collector import CalculateTensorValueFunc

from nncf.experimental.onnx.sampler import ONNXBatchSampler
from nncf.experimental.onnx.sampler import ONNXRandomBatchSampler


class ONNXLayerStatistic(LayerStatistic):
    def get_global_min_value(self, mean: bool = True):
        if mean:
            return np.mean(self.min_values)
        return np.min(self.min_values)

    def get_global_max_value(self, mean: bool = True):
        if mean:
            return np.mean(self.max_values)
        return np.max(self.max_values)


class ONNXStatisticsCollector(StatisticsCollector):
    def __init__(self, compressed_model, engine):
        super().__init__(compressed_model, engine)

    def collect_statistics(self, layers_statistics: List[LayerStatistic], num_iters: int):
        layers_to_collect_statistics = [layer_statistic.layer_name for layer_statistic in layers_statistics]
        onnx_model = self.compressed_model.original_model
        model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            # sampler = ONNXBatchSampler(self.engine.dataloader)
            sampler = ONNXRandomBatchSampler(self.engine.dataloader)
            for i, sample in enumerate(sampler):
                if i == num_iters:
                    break
                _input, target = sample
                output = self.engine.infer(_input)
                self._agregate_statistics(output, layers_statistics)
                if self.is_calculate_metric:
                    self._calculate_metric(target)

        return layers_statistics

    def _agregate_statistics(self, output, layers_statistics: List[LayerStatistic]):
        # TODO: add honest  statistics collection
        for layer_statistic in layers_statistics:
            tensor = output[layer_statistic.layer_name]
            layer_statistic.add_tensor_statistic(tensor)

    def _calculate_metric(self, target):
        pass


class ONNXTensorMinFunc(CalculateTensorValueFunc):
    @staticmethod
    def __call__(tensor: np.ndarray, axis: int):
        return np.min(tensor, axis=axis)


class ONNXTensorMaxFunc(CalculateTensorValueFunc):
    @staticmethod
    def __call__(tensor: np.ndarray, axis: int):
        return np.max(tensor, axis=axis)


def mean_min_max(x: np.ndarray):
    return np.min(x), np.max(x)
