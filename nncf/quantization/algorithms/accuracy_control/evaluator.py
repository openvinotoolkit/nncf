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
TTensor = TypeVar("TTensor")


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

    def is_metric_mode(self) -> bool:
        """
        Returns mode of `Evaluator`.

        :return: A boolean indicator where `True` means that the `Evaluator` collects
            metric value for each item and `False` means that the `Evaluator` collects
            logits for each item.
        """
        return self._metric_mode

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
        model_for_inference = self._algo_backend.prepare_for_inference(model)
        if self._metric_mode is None:
            self._determine_mode(model_for_inference, dataset)

        if not self.is_metric_mode() and indices is not None:
            raise ValueError("The `indices` parameter can be used only if Evaluator.is_metric_mode() = True")

        metric, values_for_each_item = self._validation_fn(model_for_inference, dataset.get_data(indices))

        if self.is_metric_mode():
            # This casting is necessary to cover the following cases:
            # - np.array(1.0, dtype=np.float32)
            # - np.array([1.0], dtype=np.float32)
            # - torch.tensor(1.0, dtype=torch.float32)
            # - torch.tensor([1.0], dtype=torch.float32)
            # - tf.constant(1.0, dtype=tf.float32
            # - tf.constant([1.0], dtype=tf.float32)
            values_for_each_item = [float(x) for x in values_for_each_item]

        return float(metric), values_for_each_item

    def _determine_mode(self, model_for_inference: TModel, dataset: Dataset) -> None:
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

    def collect_values_for_each_item(
        self, model: TModel, dataset: Dataset, indices: Optional[List[int]] = None
    ) -> Union[List[float], List[List[TTensor]]]:
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
                logits = engine.infer(data_item)
                values_for_each_item.append(list(logits.values()))

        return values_for_each_item
