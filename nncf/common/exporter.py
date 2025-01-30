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

from abc import ABC
from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from nncf.api.compression import TModel


class Exporter(ABC):
    """
    This is the class from which all framework-specific exporters inherit.

    An exporter is an object which provides export of the compressed model
    for deployment.
    """

    def __init__(
        self,
        model: TModel,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        model_args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Initializes an exporter.

        :param model: The model to be exported.
        :param input_names: Names to be assigned to the input tensors of the model.
        :param output_names: Names to be assigned to the output tensors of the model.
        :param model_args: Tuple of additional positional and keyword arguments
            which are required for the model's forward during export. Should be
            specified in the following format:
                - (a, b, {'x': None, 'y': y}) for positional and keyword arguments.
                - (a, b, {}) for positional arguments only.
                - ({'x': None, 'y': y},) for keyword arguments only.
        """
        self._model = model
        self._input_names = input_names
        self._output_names = output_names
        self._model_args = model_args

    @abstractmethod
    def export_model(self, save_path: str, save_format: Optional[str] = None) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format. The default format will
            be used if `save_format` is not specified.
        """
