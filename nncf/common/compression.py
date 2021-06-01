"""
 Copyright (c) 2021 Intel Corporation
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

from typing import Optional, List, Tuple, Any

from nncf.api.compression import CompressionAlgorithmController
import nncf.common.factory as factory


class BaseCompressionAlgorithmController(CompressionAlgorithmController):
    """
    Contains the implementation of the basic functionality of the compression controller.
    """

    def export_model(self,
                     save_path: str,
                     save_format: Optional[str] = None,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None,
                     model_args: Optional[Tuple[Any, ...]] = None) -> None:
        """
        Exports the compressed model to the specified format for deployment.

        Makes method-specific preparations of the model, (e.g. removing auxiliary
        layers that were used for the model compression), then exports the model to
        the specified path.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format. The default format will
            be used if `save_format` is not specified.
        :param input_names: Names to be assigned to the input tensors of the model.
        :param output_names: Names to be assigned to the output tensors of the model.
        :param model_args: Tuple of additional positional and keyword arguments
            which are required for the model's forward during export. Should be
            specified in the following format:
                - (a, b, {'x': None, 'y': y}) for positional and keyword arguments.
                - (a, b, {}) for positional arguments only.
                - ({'x': None, 'y': y},) for keyword arguments only.
        """
        self.prepare_for_export()
        exporter = factory.create_exporter(self.model, input_names, output_names, model_args)
        exporter.export_model(save_path, save_format)
