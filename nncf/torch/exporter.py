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
from typing import Any
from typing import Optional
from typing import Tuple
from functools import partial
from copy import copy
import torch

from nncf.common.exporter import Exporter
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import create_mock_tensor
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import is_tensor, get_model_device


def generate_input_names_list(num_inputs: int):
    return [f'input.{idx}' for idx in range(0, num_inputs)]


def generate_output_names_list(num_outputs: int):
    return [f'output.{idx}' for idx in range(0, num_outputs)]


def count_tensors(model_retval: Any) -> int:
    count = 0
    def counter_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal count
        count += 1
        return x

    objwalk(model_retval, is_tensor, counter_fn)
    return count


class PTExporter(Exporter):
    """
    This class provides export of the compressed model to the ONNX format.
    """

    _ONNX_FORMAT = 'onnx'
    _ONNX_DEFAULT_OPSET = 10


    @staticmethod
    def parse_format(save_format: str) -> Tuple[str, dict]:
        """
        Parse saving format to a short form and additional arguments.

        :param save_format: Saving format.

        :return
            str: short form of the save_format
            dict: additional arguments for exporter
        """
        if save_format.startswith(PTExporter._ONNX_FORMAT):
            split_format = save_format.split('_')
            opset = None

            if len(split_format) == 1:
                opset = PTExporter._ONNX_DEFAULT_OPSET
            elif len(split_format) == 2:
                opset = int(split_format[1])

            if opset is not None and opset <= 0:
                raise ValueError("Incorrect save_format, expected 'onnx' or 'onnx_<opset_version>'.")

            if opset != PTExporter._ONNX_DEFAULT_OPSET:
                nncf_logger.warning(
                    'Using {} ONNX opset version. Recommended version is {}.'.format(
                        opset, PTExporter._ONNX_DEFAULT_OPSET
                    )
                )

            return PTExporter._ONNX_FORMAT, {'opset_version': opset}
        return save_format, {}

    def export_model(self, save_path: str, save_format: Optional[str] = None) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format.
            One of the following:
                - `onnx` for export to the ONNX format.
                - `onnx_<opset_version>` for export to the ONNX format with specific opset version.
            The ONNX format will be used if `save_format` is not specified.
        """
        fn_args = {'save_path': save_path}

        if save_format is None:
            save_format = PTExporter._ONNX_FORMAT

        save_format, extra_args = PTExporter.parse_format(save_format)
        fn_args.update(extra_args)

        format_to_export_fn = {
            PTExporter._ONNX_FORMAT: self._export_to_onnx,
        }

        export_fn = format_to_export_fn.get(save_format)

        if export_fn is None:
            available_formats = list(format_to_export_fn.keys())
            raise ValueError(f'Unsupported saving format: \'{save_format}\'. '
                             f'Available formats: {available_formats}')

        export_fn(**fn_args)

    def _export_to_onnx(self, save_path: str, opset_version: int) -> None:
        """
        Exports the compressed model to the ONNX format.

        :param save_path: The path where the model will be saved.
        """
        original_device = get_model_device(self._model)
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

        if self._input_names is not None:
            input_names = self._input_names
        else:
            input_names = generate_input_names_list(len(input_tensor_list))


        # pylint:disable=unexpected-keyword-arg
        with torch.no_grad():
            # Should call this, otherwise the operations executed during export will end up in the graph.
            model.disable_dynamic_graph_building()

            if self._output_names is not None:
                output_names = self._output_names
            else:
                # Will have to run a dummy forward call in order to determine the number of outputs.
                dummy_forward = create_dummy_forward_fn(self._model.input_infos)
                retval = dummy_forward(self._model)
                output_names = generate_output_names_list(count_tensors(retval))

            torch.onnx.export(model, tuple(input_tensor_list), save_path,
                              input_names=input_names,
                              output_names=output_names,
                              enable_onnx_checker=False,
                              opset_version=opset_version,
                              # Do not fuse Conv+BN in ONNX. May cause dropout elements to appear in ONNX.
                              training=True)
            model.enable_dynamic_graph_building()
        model.forward = original_forward
        model.to(original_device)
