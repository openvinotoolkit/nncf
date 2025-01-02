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
from functools import partial
from typing import Any, Optional, Tuple

import torch
from torch.onnx import OperatorExportTypes

from nncf.common.exporter import Exporter
from nncf.common.logging import nncf_logger
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_PT_CATEGORY
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor


def generate_input_names_list(num_inputs: int):
    return [f"input.{idx}" for idx in range(0, num_inputs)]


def generate_output_names_list(num_outputs: int):
    return [f"output.{idx}" for idx in range(0, num_outputs)]


def count_tensors(obj: Any) -> int:
    count = 0

    def counter_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal count
        count += 1
        return x

    objwalk(obj, is_tensor, counter_fn)
    return count


def get_export_args(
    model: NNCFNetwork, model_args: Optional[Tuple[Any, ...]] = None, device: Optional[str] = None
) -> Tuple:
    args, kwargs = model.nncf.input_infos.get_forward_inputs(device)

    if model_args is not None:
        args = tuple(list(args) + list(model_args[:-1]))
        kwargs.update(**model_args[-1])

    def to_single_batch_tensors(obj: torch.Tensor):
        return obj[0:1]

    args = objwalk(args, is_tensor, to_single_batch_tensors)
    kwargs = objwalk(kwargs, is_tensor, to_single_batch_tensors)
    return *args, kwargs  # according to a variant of passing kwargs in torch.onnx.export doc


class PTExportFormat:
    ONNX = "onnx"


class PTExporter(Exporter):
    """
    This class provides export of the compressed model to the ONNX format.
    """

    _ONNX_DEFAULT_OPSET = 13

    @staticmethod
    def parse_format(save_format: str) -> Tuple[str, dict]:
        """
        Parse saving format to a short form and additional arguments.

        :param save_format: Saving format.

        :return
            str: short form of the save_format
            dict: additional arguments for exporter
        """
        if save_format.startswith(PTExportFormat.ONNX):
            split_format = save_format.split("_")
            opset = None

            if len(split_format) == 1:
                opset = PTExporter._ONNX_DEFAULT_OPSET
            elif len(split_format) == 2:
                opset = int(split_format[1])

            if opset is not None and opset <= 0:
                raise ValueError("Incorrect save_format, expected 'onnx' or 'onnx_<opset_version>'.")

            if opset != PTExporter._ONNX_DEFAULT_OPSET:
                nncf_logger.warning(
                    f"Exporting to ONNX opset {opset}, which is not guaranteed to work with NNCF. "
                    f"Recommended opset export version is {PTExporter._ONNX_DEFAULT_OPSET}."
                )

            return PTExportFormat.ONNX, {"opset_version": opset}
        return save_format, {}

    @tracked_function(NNCF_PT_CATEGORY, ["save_format"])
    def export_model(self, save_path: str, save_format: str = PTExportFormat.ONNX) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format.
            One of the following:
                - `onnx` for export to the ONNX format.
                - `onnx_<opset_version>` for export to the ONNX format with specific opset version.
            The ONNX format will be used if `save_format` is not specified.
        """

        fn_args = {"save_path": save_path}

        save_format, extra_args = PTExporter.parse_format(save_format)
        fn_args.update(extra_args)

        format_to_export_fn = {
            PTExportFormat.ONNX: self._export_to_onnx,
        }

        export_fn = format_to_export_fn.get(save_format)

        if export_fn is None:
            available_formats = list(format_to_export_fn.keys())
            raise ValueError(f"Unsupported saving format: '{save_format}'. Available formats: {available_formats}")

        export_fn(**fn_args)

    def _export_to_onnx(self, save_path: str, opset_version: int) -> None:
        """
        Exports the compressed model to the ONNX format.

        :param save_path: The path where the model will be saved.
        """
        original_device = get_model_device(self._model)
        model = self._model.eval().cpu()

        export_args = get_export_args(self._model, model_args=self._model_args, device="cpu")

        if self._input_names is not None:
            input_names = self._input_names
        else:
            input_names = generate_input_names_list(count_tensors(export_args))

        with torch.no_grad():
            # Should call this, otherwise the operations executed during export will end up in the graph.
            model.nncf.disable_dynamic_graph_building()

            if self._output_names is not None:
                output_names = self._output_names
            else:
                # Will have to run a dummy forward call in order to determine the number of outputs.
                dummy_forward = create_dummy_forward_fn(self._model.nncf.input_infos)
                retval = dummy_forward(self._model)
                output_names = generate_output_names_list(count_tensors(retval))

            self._torch_export_call(model, export_args, save_path, input_names, output_names, opset_version)

            model.nncf.enable_dynamic_graph_building()
        model.to(original_device)

    def _torch_export_call(self, model, input_tensor_list, save_path, input_names, output_names, opset_version):
        """
        Call of torch.onnx.export function.
        :param model: torch.nn.Module to be exported.
        :param input_tensor_list: the list containing model inputs.
        :param save_path: a string containing a path for saving onnx model.
        :param input_names: Names to be assigned to the input tensors of the model.
        :param output_names: Names to be assigned to the output tensors of the model.
        :param opset_version: the version of the onnx opset.
        """
        fn = partial(
            torch.onnx.export,
            model,
            tuple(input_tensor_list),
            save_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            training=torch.onnx.TrainingMode.EVAL,
        )
        try:
            fn()
        except torch.onnx.errors.SymbolicValueError:
            # May have failed for reasons of missing and unspecifiable shape inference
            # for quantizer ops in torch==1.13, try to export with a workaround.
            nncf_logger.warning(
                "Encountered shape inferencing failures during ONNX export. "
                "The model was exported with a workaround - some of the operations may have been exported using "
                "the `org.pytorch.aten` domain."
            )
            fn(operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
