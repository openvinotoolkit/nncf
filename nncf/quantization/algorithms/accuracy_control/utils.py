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

from typing import List, Any, Callable, Iterable

from nncf.data.dataset import Dataset
from nncf.common.factory import EngineFactory


def get_metric_for_each_item(model: Any,
                             dataset: Dataset,
                             validation_fn: Callable[[Any, Iterable[Any]], float]) -> List[float]:
    """
    Calls `validation_fn` for each item from the `dataset` and returns collected metrics.

    :param model: The model to be inferred.
    :param dataset: The dataset.
    :param validation_fn: A validation function that will be called for
        each item from the `dataset`.
    :return: A list that contains a metric for each item from the dataset.
    """
    metrics = [
        validation_fn(model, [data_item]) for data_item in dataset.get_data()
    ]
    return metrics


def get_logits_for_each_item(model: Any,
                             dataset: Dataset,
                             output_name: str) -> List[Any]:
    """
    Infers `model` for each item from the `dataset` and returns collected logits.

    :param model: The model to be inferred.
    :param dataset: The dataset.
    :param output_name: Name of output.
    :return: A list that contains logits for each item from the dataset.
    """
    engine = EngineFactory.create(model)
    outputs = [
        engine.infer(data_item)[output_name] for data_item in dataset.get_inference_data()
    ]
    return outputs
