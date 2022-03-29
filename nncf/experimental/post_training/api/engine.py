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

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import TypeVar

ModelType = TypeVar('ModelType')
TensorType = TypeVar('TensorType')
MetricType = TypeVar('MetricType')

class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """
    # TODO (Nikita Malinin): Update class with the _get_sampler() method

    def __init__(self):
        self.model = None
        self._sampler = None
        self._data_loader = None
        self._metrics = None

    # TODO (Nikita Malinin): Add statistic aggregator object (per-backend)
    
    @property
    def data_loader(self):
        return self._data_loader

    @data_loader.setter
    def data_loader(self, data_loader):
        self._data_loader = data_loader

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        self._sampler = sampler

    def set_model(self, model: ModelType) -> None:
        self.model = model

    def is_model_set(self) -> bool:
        return self.model is not None

    def transform_output(self, outputs: TensorType) -> TensorType:
        """ Processes model output data
            :param outputs: list of output data
            :return outputs: list of the output data in an order expected by the accuracy metric if any is used
        """
        return outputs

    def transform_input(self, inputs: TensorType) -> TensorType:
        """ Processes model input data based on the backend
            :param inputs: list of input data
            :return inputs: list of the input data
        """
        return inputs

    def compute_statistics(self, statistics_layout: Dict) -> Dict[str, TensorType]:
        """ Performs model inference on specified dataset subset for statistics collection and input-based layers layout
            :param statistics_layout: dictionary of stats collection functions {node_name: {stat_name: fn}}
            :return statistics: per-layer statistics for the further model optimization 
        """
        raise NotImplementedError('Method compute_statistics() should be implementer before calling!')
    
    @abstractmethod
    def compute_metrics(self, metrics_per_sample=False) -> Dict[str, MetricType]:
        """ Performs model inference on specified dataset subset for metrics calculation
            :param metrics_per_sample: whether to collect metrics for each batch
            :return metrics: a tuple of dictionaries of persample and overall metric values if 'metrics_per_sample' is True
        """
