# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

from nncf.common.factory import EngineFactory
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend

TModel = TypeVar("TModel")
TPModel = TypeVar("TPModel")
TTensor = TypeVar("TTensor")


class IterationCounter:
    """
    A wrapper for counting the passed iterations of iterable objects.
    """

    def __init__(self, iterable):
        self._iterable = iterable
        self._num_iterations = 0

    @property
    def num_iterations(self) -> int:
        return self._num_iterations

    def __iter__(self):
        self._num_iterations = 0
        for x in self._iterable:
            self._num_iterations += 1
            yield x


class Evaluator:
    """
    Evaluator encapsulates a logic to validate model and collect values for each item.
    The value is either calculated metric or model output. This is determined by the
    `Evaluator.is_metric_mode()` method.
    """

    def __init__(
        self,
        validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
        algo_backend: AccuracyControlAlgoBackend,
    ):
        """
        :param validation_fn: Validation function to validate model.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        """
        self._validation_fn = validation_fn
        self._algo_backend = algo_backend
        self._metric_mode = None
        self._num_passed_iterations = 0
        self._enable_iteration_count = False

    @property
    def num_passed_iterations(self) -> int:
        """
        Number of passed iterations during last validation process if the iteration count is enabled.

        :return: Number of passed iterations during last validation process.
        """

        return self._num_passed_iterations

    def enable_iteration_count(self) -> None:
        """
        Enable the iteration count.
        """
        self._enable_iteration_count = True

    def disable_iteration_count(self) -> None:
        """
        Disable the iteration count.
        """
        self._enable_iteration_count = False

    def is_metric_mode(self) -> bool:
        """
        Returns mode of `Evaluator`.

        :return: A boolean indicator where `True` means that the `Evaluator` collects
            metric value for each item and `False` means that the `Evaluator` collects
            logits for each item.
        """
        return self._metric_mode

    def prepare_model_for_inference(self, model: TModel) -> TPModel:
        """
        Prepares model for inference.

        :param model: A model that should be prepared.
        :return: Prepared model for inference.
        """
        return self._algo_backend.prepare_for_inference(model)

    def validate_model_for_inference(
        self, model_for_inference: TPModel, dataset: Dataset, indices: Optional[List[int]] = None
    ):
        """
        Validates prepared model for inference.

        :param model: Prepared model to validate.
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
        if self._metric_mode is None:
            self._determine_mode(model_for_inference, dataset)

        if not self.is_metric_mode() and indices is not None:
            raise ValueError("The `indices` parameter can be used only if Evaluator.is_metric_mode() = True")

        validation_dataset = dataset.get_data(indices)
        if self._enable_iteration_count:
            validation_dataset = IterationCounter(validation_dataset)

        metric, values_for_each_item = self._validation_fn(model_for_inference, validation_dataset)

        self._num_passed_iterations = validation_dataset.num_iterations if self._enable_iteration_count else 0

        if self.is_metric_mode() and values_for_each_item is not None:
            # This casting is necessary to cover the following cases:
            # - np.array(1.0, dtype=np.float32)
            # - np.array([1.0], dtype=np.float32)
            # - torch.tensor(1.0, dtype=torch.float32)
            # - torch.tensor([1.0], dtype=torch.float32)
            # - tf.constant(1.0, dtype=tf.float32
            # - tf.constant([1.0], dtype=tf.float32)
            values_for_each_item = [float(x) for x in values_for_each_item]

        return float(metric), values_for_each_item

    def validate(
        self, model: TModel, dataset: Dataset, indices: Optional[List[int]] = None
    ) -> Tuple[float, Union[None, List[float], List[List[TTensor]]]]:
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
        model_for_inference = self.prepare_model_for_inference(model)
        return self.validate_model_for_inference(model_for_inference, dataset, indices)

    def _determine_mode(self, model_for_inference: TPModel, dataset: Dataset) -> None:
        """
        Determines mode based on the type of returned value from the
        validation function.

        :param model_for_inference: Model to validate.
        :param dataset: Dataset to validate the model.
        """
        data_item = dataset.get_data([0])
        # pylint: disable=W0703
        try:
            metric_value, values_for_each_item = self._validation_fn(model_for_inference, data_item)
        except Exception:
            self._metric_mode = False

        if self._metric_mode is not None:
            return

        try:
            metric_value = metric_value if metric_value is None else float(metric_value)
        except Exception as ex:
            raise RuntimeError(
                f"Metric value of {type(metric_value)} type was returned from the `validation_fn` "
                "but the float value is expected."
            ) from ex

        convert_to_float_possible = True
        if values_for_each_item is not None:
            # pylint: disable=W0703
            try:
                _ = float(values_for_each_item[0])
            except Exception:
                convert_to_float_possible = False

        # Analyze `metric_value` and `values_for_each_item` values:
        # +--------------+----------------------+-------------+
        # | metric_value | values_for_each_item | metric_mode |
        # +--------------+----------------------+-------------+
        # | float        | None                 | True        |
        # +--------------+----------------------+-------------+
        # | float        | List[float]          | True        |
        # +--------------+----------------------+-------------+
        # | float        | List[List[TTensor]]  | False       |
        # +--------------+----------------------+-------------+
        # | None         | None                 | False       |
        # +--------------+----------------------+-------------+
        # | None         | List[float]          | UNEXPECTED  |
        # +--------------+----------------------+-------------+
        # | None         | List[List[TTensor]]  | False       |
        # +--------------+----------------------+-------------+

        self._metric_mode = False
        if isinstance(metric_value, float) and (values_for_each_item is None or convert_to_float_possible):
            self._metric_mode = True
        elif values_for_each_item is not None and not isinstance(values_for_each_item[0], list):
            raise RuntimeError("Unexpected return value from provided validation function.")

    def collect_values_for_each_item_using_model_for_inference(
        self, model_for_inference: TPModel, dataset: Dataset, indices: Optional[List[int]] = None
    ) -> Union[List[float], List[List[TTensor]]]:
        """
        Collects value for each item from the dataset using prepared model for inference.
        If `is_metric_mode()` returns `True` then i-th value is a metric for i-th data item.
        It is an output of the model for i-th data item otherwise.

        :param model: Model to infer.
        :param dataset: Dataset to collect values.
        :param indices: The zero-based indices of data items that should be selected from
            the dataset.
        :return: Collected values.
        """
        if self._metric_mode:
            # Collect metrics for each item
            values_for_each_item = [
                self._validation_fn(model_for_inference, [data_item])[0] for data_item in dataset.get_data(indices)
            ]
        else:
            # Collect outputs for each item
            engine = EngineFactory.create(model_for_inference)

            values_for_each_item = []
            for data_item in dataset.get_inference_data(indices):
                logits = engine.infer(data_item)
                values_for_each_item.append(list(logits.values()))

        self._num_passed_iterations = len(values_for_each_item) if self._enable_iteration_count else 0

        return values_for_each_item

    def collect_values_for_each_item(
        self, model: TModel, dataset: Dataset, indices: Optional[List[int]] = None
    ) -> Union[List[float], List[List[TTensor]]]:
        """
        Collects value for each item from the dataset. If `is_metric_mode()`
        returns `True` then i-th value is a metric for i-th data item. It
        is an output of the model for i-th data item otherwise.

        :param model: A target model.
        :param dataset: Dataset to collect values.
        :param indices: The zero-based indices of data items that should be selected from
            the dataset.
        :return: Collected values.
        """
        model_for_inference = self.prepare_model_for_inference(model)
        return self.collect_values_for_each_item_using_model_for_inference(model_for_inference, dataset, indices)
