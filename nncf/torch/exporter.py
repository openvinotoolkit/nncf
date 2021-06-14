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

from typing import Optional
from functools import partial
from copy import copy
import torch

from nncf.common.exporter import Exporter
from nncf.torch.dynamic_graph.graph_tracer import create_mock_tensor


class PTExporter(Exporter):
    """
    This class provides export of the compressed model to the ONNX format.
    """

    _ONNX_FORMAT = 'onnx'

    def export_model(self, save_path: str, save_format: Optional[str] = None) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format.
            One of the following:
                - `onnx` for export to the ONNX format.
            The ONNX format will be used if `save_format` is not specified.
        """
        if save_format is None:
            save_format = PTExporter._ONNX_FORMAT

        format_to_export_fn = {
            PTExporter._ONNX_FORMAT: self._export_to_onnx,
        }

        export_fn = format_to_export_fn.get(save_format)

        if export_fn is None:
            available_formats = list(format_to_export_fn.keys())
            raise ValueError(f'Unsupported saving format: \'{save_format}\'. '
                             f'Available formats: {available_formats}')

        export_fn(save_path)

    def _export_to_onnx(self, save_path: str) -> None:
        """
        Exports the compressed model to the ONNX format.

        :param save_path: The path where the model will be saved.
        """
        model = self._model.eval().cpu()
        input_tensor_list = []
        for info in self._model.input_infos:
            single_batch_info = copy(info)
            input_shape = tuple([1] + list(info.shape)[1:])
            single_batch_info.shape = input_shape
            input_tensor_list.append(create_mock_tensor(single_batch_info, 'cpu'))

        original_forward = model.forward
        args = self._model_args[:-1]
        kwargs = self._model_args[-1]
        model.forward = partial(model.forward, *args, **kwargs)

        # pylint:disable=unexpected-keyword-arg
        with torch.no_grad():
            # Should call this, otherwise the operations executed during export will end up in graph.
            model.disable_dynamic_graph_building()
            torch.onnx.export(model, tuple(input_tensor_list), save_path,
                              input_names=self._input_names,
                              output_names=self._output_names,
                              enable_onnx_checker=False,
                              opset_version=10,
                              # Do not fuse Conv+BN in ONNX. May cause dropout elements to appear in ONNX.
                              training=True)
            model.enable_dynamic_graph_building()
        model.forward = original_forward
