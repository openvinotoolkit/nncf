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

import numpy as np
import onnx

from nncf.data import Dataset
from nncf.parameters import convert_ignored_scope_to_list
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.common.quantization.structs import QuantizationPreset
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.post_training.api.dataset import Dataset as PTQDataset
from nncf.experimental.post_training.compression_builder import CompressionBuilder


# TODO(alexsu52): It is a workaround and should be removed.
# ONNX expects the dataset instance with following implemented interfaces 
# https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/post_training/api/dataset.py. 
# Thus the intermediate class is needed to run quantization in ONNX backend.
class ONNXDataset(PTQDataset):
    """
    This class wraps the nncf.Dataset.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, dataset: Dataset):
        super().__init__(shuffle=False, has_batch_dim=False)
        self._dataset = dataset
        self._length = None
        self._it = None
        self._elem = None
        self._elem_idx = -1

    def __len__(self) -> int:
        if self._length is None:
            data = self._dataset.get_inference_data()
            self._length = ONNXDataset._get_length(data)
        return self._length

    def __getitem__(self, index: int):
        if index == self._elem_idx:
            return self._elem

        if self._it is None or index < self._elem_idx:
            self._it = iter(self._dataset.get_inference_data())
            self._elem_idx = -1

        while self._elem_idx != index:
            try:
                self._elem = next(self._it)
                self._elem_idx = self._elem_idx + 1
            except StopIteration:
                self._it = None
                self._elem = None
                self._elem_idx = -1
                break

        if self._elem is None:
            raise IndexError('Index out of range.')

        item = self._elem
        if isinstance(item, dict):
            for key in item:
                if not isinstance(item[key], np.ndarray):
                    raise RuntimeError('The input tensor should be numpy ndarray')
                item[key] = ONNXNNCFTensor(item[key])
        return item

    @staticmethod
    def _get_length(iterable) -> int:
        length = 0
        for _ in iterable:
            length = length + 1

        return length


def quantize_impl(model: onnx.ModelProto,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None) -> onnx.ModelProto:
    """
    Implementation of the `quantize()` method for the ONNX backend.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')
    if fast_bias_correction is False:
        raise ValueError(f'fast_bias_correction={fast_bias_correction} is not '
                          'supported')
    if ignored_scope is not None and ignored_scope.types is not None:
        raise RuntimeError('Quantization algorithm form the ONNX backend '
                           'does not support operation types in the ignored '
                           'scopes yet')

    builder = CompressionBuilder()

    quantization_parameters = PostTrainingQuantizationParameters(
        preset=preset,
        target_device=target_device,
        number_samples=subset_size,
        ignored_scopes=convert_ignored_scope_to_list(ignored_scope)
    )

    quantization = PostTrainingQuantization(quantization_parameters)
    builder.add_algorithm(quantization)

    onnx_dataset = ONNXDataset(calibration_dataset)
    quantized_model = builder.apply(model, onnx_dataset)

    return quantized_model
