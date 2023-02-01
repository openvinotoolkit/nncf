"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Optional
from typing import Callable
from typing import Iterable
from typing import Any
from typing import List
from typing import Dict

from openvino.tools import pot
import openvino.runtime as ov

from nncf.api.compression import TModel
from nncf.data import Dataset
from nncf.data.dataset import DataProvider


# TODO(andrey-churkin): This class should be removed after refactoring of the OVEngine class.
# We will be able to do that when we will have moved the POT code in the NNCF.
class DummyMetric:
    """
    Dummy implementation of the pot.Metric abstract class. It is used for compatibility
    with the POT API. All docstring for the methods can be
    found [here](https://docs.openvino.ai/latest/pot_compression_api_README.html#metric).
    """

    def __init__(self):
        self._avg_value = None
        self.name = 'original_metric'

    @property
    def avg_value(self):
        return {self.name: self._avg_value}

    @avg_value.setter
    def avg_value(self, value):
        self._avg_value = value

    def get_attributes(self):
        attributes = {
            self.name: {
                'direction': 'higher-better',
                'type': self.name,
            }
        }
        return attributes

    def reset(self):
        self.avg_value = None


# TODO(andrey-churkin): The algorithms from the POT use the `__len__()` method.
# This class should be removed when we change all algorithms.
class DatasetWrapper:
    def __init__(self, iterable):
        self._iterable = iterable
        self._length = None

    def __iter__(self):
        for idx, x in enumerate(self._iterable):
            yield (idx, x)

    # pylint: disable=protected-access
    def __len__(self) -> int:
        if self._length is None:
            data_source = None
            if isinstance(self._iterable, DataProvider):
                indices = self._iterable._indices
                if indices:
                    self._length = len(indices)
                    return self._length
                data_source = self._iterable._data_source
            iterable = data_source if data_source else self._iterable
            self._length = DatasetWrapper._get_length(iterable)
        return self._length

    @staticmethod
    def _get_length(iterable) -> int:
        try:
            length = len(iterable)
            return length
        except (TypeError, AttributeError):
            length = None

        length = 0
        for _ in iterable:
            length = length + 1
        return length


def calc_per_sample_metrics(compiled_model: ov.CompiledModel,
                            val_func: Callable[[ov.CompiledModel, Iterable[Any]], float],
                            dataset: Dataset,
                            subset_indices: List[int]) -> List[Dict[str, Any]]:
    per_sample_metrics = []
    for inputs in dataset.get_data(subset_indices):
        value = val_func(compiled_model, [inputs])
        per_sample_metrics.append({
            'sample_id': len(per_sample_metrics),
            'metric_name': 'original_metric',
            'result': value
        })
    return per_sample_metrics


# TODO(andrey-churkin): This class should be refactored. We will be able to do that when
# we will have POT code in NNCF.
class OVEngine(pot.IEEngine):
    """
    Implementation of the engine for OpenVINO backend.

    All docstring for the methods can be found
    [here](https://docs.openvino.ai/latest/pot_compression_api_README.html#engine).
    """

    def __init__(self,
                 config,
                 calibration_dataset: Dataset,
                 validation_dataset: Dataset,
                 validation_fn: Optional[Callable[[TModel, Iterable[Any]], float]] = None,
                 use_original_metric: bool = False):
        metric = DummyMetric() if validation_fn else None
        super().__init__(config, DatasetWrapper(validation_dataset.get_inference_data()), metric)
        self._calibration_dataset = calibration_dataset  # TODO(andrey-churkin): Not used now.
        self._validation_dataset = validation_dataset
        self._validation_fn = validation_fn
        self.use_original_metric = use_original_metric

    def predict(self,
                stats_layout=None,
                sampler=None,
                stat_aliases=None,
                metric_per_sample=False,
                print_progress=False):

        if self._model is None:
            raise Exception('Model was not set in Engine class')

        subset_indices = None
        if sampler:
            subset_indices = sorted(getattr(sampler, '_subset_indices'))

        is_full_dataset = subset_indices is None or len(subset_indices) == len(self.data_loader)
        if self._validation_fn and (is_full_dataset or self.use_original_metric):
            compiled_model = self._ie.compile_model(model=self._model, device_name=self._device)
            self._metric.avg_value = self._validation_fn(compiled_model,
                                                         self._validation_dataset.get_data(subset_indices))
            if not metric_per_sample and stats_layout is None:
                metrics = self._metric.avg_value
                self._reset()
                return metrics, {}

        if self.use_original_metric and metric_per_sample:
            self._per_sample_metrics = self.calculate_per_sample_metrics(subset_indices)
            if stats_layout is None:
                metrics = self._metric.avg_value
                metrics = (sorted(self._per_sample_metrics, key=lambda i: i['sample_id']), metrics)
                self._reset()
                return metrics, {}

        dataset_wrapper = DatasetWrapper(self._validation_dataset.get_inference_data(subset_indices))
        return super().predict(stats_layout, dataset_wrapper, stat_aliases, metric_per_sample, print_progress)

    def calculate_per_sample_metrics(self, subset_indices: List[int]):
        if subset_indices is None:
            subset_indices = list(range(len(self.data_loader)))

        if self.use_original_metric:
            compiled_model = self._ie.compile_model(model=self._model, device_name=self._device)
            per_sample_metrics = calc_per_sample_metrics(compiled_model,
                                                         self._validation_fn,
                                                         self._validation_dataset,
                                                         subset_indices)
        else:
            dataset_wrapper = DatasetWrapper(self._validation_dataset.get_inference_data(subset_indices))
            ans = super().predict(None, dataset_wrapper, None, True, None)
            (per_sample_metrics, _), *_  = ans
        return per_sample_metrics

    def _update_metrics(self, output, annotations, need_metrics_per_sample: bool = False):
        if need_metrics_per_sample and not self.use_original_metric:
            sample_id, _ = annotations[0]
            self._per_sample_metrics.append({
                    'sample_id': sample_id,
                    'result': output[0],
                    'metric_name': 'nmse'
            })

    @staticmethod
    def _process_batch(batch):
        index, input_data = batch
        return [(index, None)], [input_data], None
