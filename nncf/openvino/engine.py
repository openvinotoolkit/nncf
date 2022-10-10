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
from time import time

from openvino.runtime import AsyncInferQueue
from openvino.tools import pot

from nncf.api.compression import ModelType
from nncf.data import DataLoader
from nncf.data.utils import create_subset


logger = pot.utils.logger.get_logger(__name__)


class DummyMetric(pot.Metric):
    def __init__(self, higher_better: bool = True):
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


def _pot_sampler_to_nncf_dataloader(sampler) -> DataLoader:
    dataloader = sampler._data_loader
    indices = sampler._subset_indices

    if not isinstance(dataloader, DataLoader):
        raise Exception(f'Unexpected type of data loader: {type(dataloader)}')

    if isinstance(sampler, pot.samplers.batch_sampler.BatchSampler):
        return create_subset(dataloader, indices)

    raise Exception(f'Unexpected type of sampler: {type(sampler)}')


class OVEngine(pot.IEEngine):
    def __init__(self,
                 config,
                 calibration_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 validation_fn: Optional[Callable[[ModelType, DataLoader], float]] = None):
        metric = DummyMetric() if validation_fn is not None else None
        super().__init__(config, validation_dataloader, metric)
        self._calibration_dataloader = calibration_dataloader  #TODO(andrey-churkin): Not used now.
        self._validation_dataloader = validation_dataloader
        self._validation_fn = validation_fn

    @property
    def data_loader(self):
        return self._validation_dataloader

    def _process_dataset(self,
                         stats_layout,
                         sampler,
                         print_progress=False,
                         need_metrics_per_sample=False):
        compiled_model = self._ie.compile_model(self._model, self._device)
        infer_request = compiled_model.create_infer_request()
        dataloader = _pot_sampler_to_nncf_dataloader(sampler)

        for data in dataloader:
            input_data = dataloader.transform_fn(data)
            outputs = infer_request.infer(self._fill_input(compiled_model, input_data))
            self._process_infer_output(stats_layout, outputs, None, None, need_metrics_per_sample)

    def _process_dataset_async(self,
                               stats_layout,
                               sampler,
                               print_progress=False,
                               need_metrics_per_sample=False,
                               requests_num=0):
        total_length = len(sampler)
        dataloader = _pot_sampler_to_nncf_dataloader(sampler)

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

        dataloader_iter = iter(enumerate(dataloader))
        # Start inference
        start_time = time()
        infer_queue.set_callback(completion_callback)
        for batch_id, data in dataloader_iter:
            input_data = dataloader.transform_fn(data)
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
            if sampler is None:
                dataloader = self._data_loader
            else:
                dataloader = _pot_sampler_to_nncf_dataloader(sampler)
            self._metric.avg_value = self._validation_fn(self._model, dataloader)

        return super().predict(stats_layout, sampler, stat_aliases, metric_per_sample, print_progress)
