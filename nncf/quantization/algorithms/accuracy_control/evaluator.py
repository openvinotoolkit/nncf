# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.backend import PreparedModel

TModel = TypeVar("TModel")
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


@dataclass
class MetricResults:
    """
    Results of metrics collection.

    :param metric_value: Aggregated metric value.
    :param values_for_each_item: Metric values for each data item.
    :param preparation_time: Time that it takes to prepare model for validation.
    :param validation_time: Time that it takes to validate model.
    """

    metric_value: float
    values_for_each_item: Union[List[float], List[List[TTensor]]]
    preparation_time: float
    validation_time: float


class Evaluator:
    """
    Evaluator encapsulates a logic to validate model and collect values for each item.
    The value is either calculated metric or model output. This is determined by the
    `Evaluator.is_metric_mode()` method.
    """

    def __init__(
        self, validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]]
    ):
        """
        :param validation_fn: Validation function to validate model.
        """
        self._validation_fn = validation_fn
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

    def is_metric_mode(self) -> Optional[bool]:
        """
        Returns mode of `Evaluator`.

        :return: A boolean indicator where `True` means that the `Evaluator` collects
            metric value for each item and `False` means that the `Evaluator` collects
            logits for each item.
        """
        return self._metric_mode

    def prepare_model(self, model: TModel) -> PreparedModel:
        """
        Prepares model for inference.

        :param model: A model that should be prepared.
        :return: Prepared model for inference.
        """
        backend = get_backend(model)

        if backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVPreparedModel

            return OVPreparedModel(model)

        if backend == BackendType.ONNX:
            from nncf.quantization.algorithms.accuracy_control.onnx_backend import ONNXPreparedModel

            return ONNXPreparedModel(model)

        raise NotImplementedError(
            f"The `prepare_model_for_inference()` method is not implemented for the {backend} backend."
        )

    def validate_prepared_model(
        self, prepared_model: PreparedModel, dataset: Dataset, indices: Optional[List[int]] = None
    ):
        """
        Validates prepared model for inference.

        :param prepared_model: Prepared model to validate.
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
            self._metric_mode = Evaluator.determine_mode(prepared_model, dataset, self._validation_fn)

        if not self.is_metric_mode() and indices is not None:
            raise ValueError("The `indices` parameter can be used only if Evaluator.is_metric_mode() = True")

        validation_dataset = dataset.get_data(indices)
        if self._enable_iteration_count:
            validation_dataset = IterationCounter(validation_dataset)

        metric, values_for_each_item = self._validation_fn(prepared_model.model_for_inference, validation_dataset)

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
        prepared_model = self.prepare_model(model)
        return self.validate_prepared_model(prepared_model, dataset, indices)

    @staticmethod
    def determine_mode(
        prepared_model: PreparedModel,
        dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    ) -> bool:
        """
        Determines mode based on the type of returned value from the
        validation function.

        :param prepared_model: Model to validate.
        :param dataset: Dataset to validate the model.
        :param validation_fn: Validation function to validate model.
        :return: A boolean indicator where `True` means that the `Evaluator` collects
            metric value for each item and `False` means that the `Evaluator` collects
            logits for each item.
        """
        metric_mode = None

        data_item = dataset.get_data([0])

        try:
            metric_value, values_for_each_item = validation_fn(prepared_model.model_for_inference, data_item)
        except Exception:
            metric_mode = False

        if metric_mode is not None:
            return metric_mode

        try:
            metric_value = metric_value if metric_value is None else float(metric_value)
        except Exception as ex:
            raise nncf.InternalError(
                f"Metric value of {type(metric_value)} type was returned from the `validation_fn` "
                "but the float value is expected."
            ) from ex

        convert_to_float_possible = True
        if values_for_each_item is not None:
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

        metric_mode = False
        if isinstance(metric_value, float) and (values_for_each_item is None or convert_to_float_possible):
            metric_mode = True
        elif values_for_each_item is not None and not isinstance(values_for_each_item[0], list):
            raise nncf.InternalError("Unexpected return value from provided validation function.")

        return metric_mode

    def collect_values_for_each_item_using_prepared_model(
        self, prepared_model: PreparedModel, dataset: Dataset, indices: Optional[List[int]] = None
    ) -> Union[List[float], List[List[TTensor]]]:
        """
        Collects value for each item from the dataset using prepared model for inference.
        If `is_metric_mode()` returns `True` then i-th value is a metric for i-th data item.
        It is an output of the model for i-th data item otherwise.

        :param prepared_model: Model to infer.
        :param dataset: Dataset to collect values.
        :param indices: The zero-based indices of data items that should be selected from the dataset.
        :return: Collected values.
        """
        total = len(indices) if indices is not None else dataset.get_length()
        if self._metric_mode:
            # Collect metrics for each item
            values_for_each_item = [
                self._validation_fn(prepared_model.model_for_inference, [data_item])[0]
                for data_item in track(dataset.get_data(indices), total=total, description="Collecting metrics")
            ]
        else:
            # Collect outputs for each item
            values_for_each_item = []
            for data_item in track(dataset.get_inference_data(indices), total=total, description="Collecting outputs"):
                logits = prepared_model(data_item)
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
        :param indices: The zero-based indices of data items that should be selected from the dataset.
        :return: Collected values.
        """
        prepared_model = self.prepare_model(model)
        return self.collect_values_for_each_item_using_prepared_model(prepared_model, dataset, indices)

    def collect_metric_results(self, model: TModel, dataset: Dataset, model_name: str = "") -> MetricResults:
        """
        Collects metric results.

        :param model: Input model.
        :param dataset: Dataset used to collect metrics.
        :param model_name: Model name.
        :return: Collected metric results.
        """
        nncf_logger.info(f"Validation of {model_name} model was started")

        with timer() as preparation_time:
            prepared_model = self.prepare_model(model)

        with timer() as validation_time:
            metric, values_for_each_item = self.validate_prepared_model(prepared_model, dataset)

        nncf_logger.info(f"Metric of {model_name} model: {metric}")

        if values_for_each_item is None:
            nncf_logger.info(f"Collecting values for each data item using the {model_name} model")
            with timer():
                values_for_each_item = self.collect_values_for_each_item_using_prepared_model(prepared_model, dataset)

        return MetricResults(metric, values_for_each_item, preparation_time(), validation_time())
