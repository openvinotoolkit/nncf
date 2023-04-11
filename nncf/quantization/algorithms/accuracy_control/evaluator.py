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

from typing import Any, Callable, Iterable, Tuple, Union, List, TypeVar, Optional

from nncf.data.dataset import Dataset
from nncf.common.factory import EngineFactory
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend


TModel = TypeVar('TModel')


class Output:
    """
    Contains ordered list of raw model outputs.
    """

    def __init__(self):
        self._outputs = []

    def __iter__(self):
        return iter(self._outputs)

    def register(self, value: Any) -> None:
        """
        Registers `value` as a raw model output.
        """
        self._outputs.append(value)


class Evaluator:
    """
    Evaluator encapsulate a logic to validate model and collect values for each item.
    The value is either calculated metric or model output. This is determined by the
    `Evaluator.is_metric_mode()` method.
    """

    def __init__(self,
                 validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[List[float], List[Output], None]]],
                 algo_backend: AccuracyControlAlgoBackend):
        """
        :param validation_fn: Validation function to validate model.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        """
        self._validation_fn = validation_fn
        self._algo_backend = algo_backend
        # If True, the second element in returned value from
        # validate() method is List[float] i.e. list of metrics
        # for each item. If False, the second element is List[Output]
        # i.e. list of logits for each item.
        self._metric_mode = None

    def is_metric_mode(self) -> bool:
        """
        Returns mode of `Evaluator`.
        :return: A boolean indicator where `True` means that the `Evaluator` collects
            metric value for each item and `False` means that the `Evaluator` collects
            logits for each item.
        """
        return self._metric_mode

    def validate(self,
                 model: TModel,
                 dataset: Dataset,
                 indices: Optional[List[int]] = None) -> Tuple[float, Union[List[float], List[Output], None]]:
        """
        Validates model.

        :param model: Model to validate.
        :param dataset: Dataset to validate the model.
        :param indices: Zero-based indices of data items that should be selected from
            the dataset.
        :return: A tuple (metric_value, values_for_each_item) where
            - metric_values: This is a metric for the model.
            - values_for_each_item: If the `Evaluator.is_metric_mode()` condition is true,
                then `values_for_each_item` represents the list of metric value for each item.
                Otherwise, if the condition is false, it represents list of logits for each
                item.
        """
        model_for_inference = self._algo_backend.prepare_for_inference(model)
        if self._metric_mode is None:
            self._determine_mode(model_for_inference, dataset)

        metric, values_for_each_item = self._validation_fn(model_for_inference, dataset.get_data(indices))

        return metric, values_for_each_item

    def _determine_mode(self, model_for_inference: TModel, dataset: Dataset) -> None:
        """
        Determines mode based on the type of returned value from the
        validation function.
        :param model_for_inference: Model to validate.
        :param dataset: Dataset to validate the model.
        """
        data_item = dataset.get_data([0])
        try:
            _, values_for_each_item = self._validation_fn(model_for_inference, data_item)

            if values_for_each_item is None or isinstance(values_for_each_item[0], float):
                self._metric_mode = True
            elif isinstance(values_for_each_item[0], Output):
                self._metric_mode = False
            else:
                raise RuntimeError('Unexpected return value from provided validation function.')
        except Exception:
            self._metric_mode = False

    def collect_values_for_each_item(self,
                                     model: Any,
                                     dataset: Dataset,
                                     indices: Optional[List[int]] = None) -> Union[List[float], List[Output]]:
        """
        Collects value for each item from the dataset. If `is_metric_mode()`
        returns `True` then i-th value is a metric for i-th data item. It
        is an output of the model for i-th data item otherwise.
        :param model: Model to infer.
        :param dataset: Dataset to collect values.
        :param indices: The zero-based indices of data items that should be selected from
            the dataset.
        :return: Collected values.
        """
        if self._metric_mode:
            # Collect metrics for each item
            model_for_inference = self._algo_backend.prepare_for_inference(model)
            values_for_each_item = [
                self._validation_fn(model_for_inference, [data_item]) for data_item in dataset.get_data(indices)
            ]
        else:
            # Collect outputs for each item
            engine = EngineFactory.create(model)

            values_for_each_item = []
            for data_item in dataset.get_inference_data(indices):
                output = Output()
                logits = engine.infer(data_item)
                for x in logits.values():
                    output.register(x)
                values_for_each_item.append(output)

        return values_for_each_item

