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

import multiprocessing
from typing import Optional, Iterable
from collections import OrderedDict

import numpy as np
import openvino.runtime as ov
import nncf
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
from nncf.experimental.openvino.quantization.quantize import \
    quantize_with_accuracy_control as pot_quantize_with_native_accuracy_control
from nncf.experimental.openvino_native.quantization.quantize import \
    quantize_with_accuracy_control as native_quantize_with_native_accuracy_control


class ACValidationFunction:
    """
    Implementation of a validation function using the Accuracy Checker.
    """

    def __init__(self,
                 model_evaluator: ModelEvaluator,
                 metric_name: str,
                 requests_number: Optional[int] = None):
        """
        :param model_evaluator: Model Evaluator.
        :param metric_name: Name of a metric.
        :param requests_number: A number of infer requests. If it is `None`,
            the count will be selected automatically.
        """
        self._model_evaluator = model_evaluator
        self._metric_name = metric_name
        self._requests_number = requests_number

    def __call__(self, compiled_model: ov.CompiledModel, indices: Optional[Iterable[int]] = None) -> float:
        """
        Calculates metrics for the provided model.

        :param compiled_model: A compiled model to validate.
        :param indices: The zero-based indices of data items
            that should be selected from the whole dataset.
        :return: Calculated metrics.
        """
        self._model_evaluator.launcher.exec_network = compiled_model
        self._model_evaluator.launcher.infer_request = compiled_model.create_infer_request()

        if indices:
            indices = list(indices)

        kwargs = {
            'subset': indices,
            'check_progress': False,
            'dataset_tag': '',
            'calculate_metrics': True,
        }

        if self._requests_number == 1:
            self._model_evaluator.process_dataset(**kwargs)
        else:
            self._set_requests_number(kwargs, self._requests_number)
            self._model_evaluator.process_dataset_async(**kwargs)

        # Calculate metrics
        metrics = OrderedDict([
            (
                metric.name, np.mean(metric.evaluated_value)
                    if metric.meta.get('calculate_mean', True) else metric.evaluated_value[0]
            )
            for metric in self._model_evaluator.compute_metrics(print_results=False)
        ])

        self._model_evaluator.reset()

        return metrics[self._metric_name]

    @staticmethod
    def _set_requests_number(params, requests_number):
        if requests_number:
            params['nreq'] = np.clip(requests_number, 1, multiprocessing.cpu_count())
            if params['nreq'] != requests_number:
                print('Number of requests {} is out of range [1, {}]. Will be used {}.'
                      .format(requests_number, multiprocessing.cpu_count(), params['nreq']))


def quantize_model_with_accuracy_control(xml_path: str,
                                         bin_path: str,
                                         accuracy_checcker_config,
                                         quantization_impl: str,
                                         quantization_parameters):
    ov_model = ov.Core().read_model(xml_path, bin_path)
    model_evaluator = create_model_evaluator(accuracy_checcker_config)
    model_evaluator.load_network([{'model': ov_model}])
    model_evaluator.select_dataset('')

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(
            batch_input, batch_annotation)
        return filled_inputs[0]

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    validation_dataset = nncf.Dataset(list(range(model_evaluator.dataset.full_size)))

    metric_name = accuracy_checcker_config['models'][0]['datasets'][0]['metrics'][0].get('name', None)
    if metric_name is None:
        metric_name = accuracy_checcker_config['models'][0]['datasets'][0]['metrics'][0]['type']
    validation_fn = ACValidationFunction(model_evaluator, metric_name)

    name_to_quantization_impl_map = {
        'pot': pot_quantize_with_native_accuracy_control,
        'native': native_quantize_with_native_accuracy_control,
    }

    quantization_impl_fn = name_to_quantization_impl_map.get(quantization_impl)
    if quantization_impl:
        quantized_model = quantization_impl_fn(ov_model, calibration_dataset, validation_dataset,
                                               validation_fn, **quantization_parameters)
    else:
        raise NotImplementedError(f'Unsupported implementation: {quantization_impl}')

    return quantized_model
