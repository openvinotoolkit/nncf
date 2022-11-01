"""
 Copyright (c) 2022 Intel Corporation
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
from time import time

from openvino.runtime import AsyncInferQueue
from openvino.tools import pot

from nncf.api.compression import T_model
from nncf.data import Dataset


logger = pot.utils.logger.get_logger(__name__)


# TODO(andrey-churkin): This class should be removed after refactoring of the OVEngine class.
# We will be able to do that when we will have moved the POT code in the NNCF.
class DummyMetric(pot.Metric):
    """
    Dummy implementation of the pot.Metric abstract class. It is used for compatibility
    with the POT API. All docstring for the methods can be
    found [here](https://docs.openvino.ai/latest/pot_compression_api_README.html#metric).
    """

    def __init__(self, higher_better: bool = True):
        super().__init__()
        self._name = 'custom_metric'
        self._higher_better = higher_better
        self._avg_value = None

    @property
    def higher_better(self):
        return self._higher_better

    @property
    def avg_value(self):
        return {self._name: self._avg_value}

    @avg_value.setter
    def avg_value(self, value):
        self._avg_value = value

    @property
    def value(self):
        raise NotImplementedError()

    def get_attributes(self):
        attributes = {
            self._name: {
                'direction': 'higher-better' if self.higher_better else 'higher-worse',
                'type': 'user_type',
            }
        }
        return attributes

    def update(self, output, target):
        raise NotImplementedError()

    def reset(self):
        self._avg_value = None


# TODO(andrey-churkin): This class should be refactored. We will be able to do that when
# we will have moved the POT code in the NNCF.
class OVEngine(pot.IEEngine):
    """
    Implementation of the engine for the OpenVINO backend.

    All docstring for the methods can be found
    [here](https://docs.openvino.ai/latest/pot_compression_api_README.html#engine).
    """

    def __init__(self,
                 config,
                 calibration_dataset: Dataset,
                 validation_dataset: Dataset,
                 validation_fn: Optional[Callable[[T_model, Iterable[Any]], float]] = None):
        metric = DummyMetric() if validation_fn is not None else None
        super().__init__(config, validation_dataset, metric)
        self._calibration_dataset = calibration_dataset  # TODO(andrey-churkin): Not used now.
        self._validation_dataset = validation_dataset
        self._validation_fn = validation_fn

    @property
    def data_loader(self):
        return self._validation_dataset

    def _process_dataset(self,
                         stats_layout,
                         sampler,
                         print_progress=False,
                         need_metrics_per_sample=False):
        compiled_model = self._ie.compile_model(self._model, self._device)
        infer_request = compiled_model.create_infer_request()

        indices = getattr(sampler, '_subset_indices')
        for input_data in self._validation_dataset.get_inference_data(indices):
            outputs = infer_request.infer(self._fill_input(compiled_model, input_data))
            self._process_infer_output(stats_layout, outputs, None, None, need_metrics_per_sample)

    def _process_dataset_async(self,
                               stats_layout,
                               sampler,
                               print_progress=False,
                               need_metrics_per_sample=False,
                               requests_num=0):
        total_length = len(sampler)
        indices = getattr(sampler, '_subset_indices')

        def completion_callback(request, user_data):
            start_time, batch_id = user_data
            self._process_infer_output(stats_layout, request.results, None, None, need_metrics_per_sample)

            # Print progress
            if self._print_inference_progress(progress_log_fn, batch_id, total_length, start_time, time()):
                start_time = time()

        progress_log_fn = logger.info if print_progress else logger.debug
        self._ie.set_property(self._device,
                              {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO', 'CPU_BIND_THREAD': 'YES'})
        # Load model to the plugin
        compiled_model = self._ie.compile_model(model=self._model, device_name=self._device)
        optimal_requests_num = compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
        requests_num = optimal_requests_num if requests_num == 0 else requests_num
        logger.debug('Async mode requests number: %d', requests_num)
        infer_queue = AsyncInferQueue(compiled_model, requests_num)

        progress_log_fn('Start inference of %d images', total_length)

        dataloader_iter = iter(enumerate(self._validation_dataset.get_inference_data(indices)))
        # Start inference
        start_time = time()
        infer_queue.set_callback(completion_callback)
        for batch_id, input_data in dataloader_iter:
            user_data = (start_time, batch_id)
            infer_queue.start_async(self._fill_input(compiled_model, input_data), user_data)
        infer_queue.wait_all()
        progress_log_fn('Inference finished')

    def _process_infer_output(self,
                              stats_layout,
                              predictions,
                              batch_annotations,
                              batch_meta,
                              need_metrics_per_sample):
        if stats_layout:
            self._collect_statistics(predictions, stats_layout)

        processed_outputs = pot.engines.utils.process_raw_output(predictions)
        outputs = {name: processed_outputs[name] for name in self._output_layers}
        logits = self.postprocess_output(outputs, None)

        if need_metrics_per_sample:
            self._per_sample_metrics.append(
                {
                    'sample_id': len(self._per_sample_metrics),
                    'metric_name': 'user_metric',
                    'result': logits
                }
            )

    def predict(self,
                stats_layout=None,
                sampler=None,
                stat_aliases=None,
                metric_per_sample=False,
                print_progress=False):
        if self._model is None:
            raise Exception('Model was not set in Engine class')

        if self._validation_fn is not None:
            indices = None
            if sampler:
                indices = getattr(sampler, '_subset_indices')
            self._metric.avg_value = self._validation_fn(self._model, self._validation_dataset.get_data(indices))

        return super().predict(stats_layout, sampler, stat_aliases, metric_per_sample, print_progress)
