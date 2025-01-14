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
from pathlib import Path

import torch

from examples.common.sample_config import SampleConfig
from examples.torch.common.example_logger import logger
from nncf.api.compression import CompressionAlgorithmController
from nncf.torch.exporter import count_tensors
from nncf.torch.exporter import generate_input_names_list
from nncf.torch.exporter import get_export_args


def export_model(ctrl: CompressionAlgorithmController, config: SampleConfig) -> None:
    """
    Export compressed model ot OpenVINO format.

    :param controller: The compression controller.
    :param config: The sample config.
    """
    model = ctrl.model if config.no_strip_on_export else ctrl.strip()
    model = model.eval().cpu()

    export_args = get_export_args(model, device="cpu")
    input_names = generate_input_names_list(count_tensors(export_args))

    input_tensor_list = []
    input_shape_list = []
    for info in model.nncf.input_infos.elements:
        input_shape = tuple([1] + info.shape[1:])
        input_tensor_list.append(torch.rand(input_shape))
        input_shape_list.append(input_shape)

    if len(input_tensor_list) == 1:
        input_tensor_list = input_tensor_list[0]
        input_shape_list = input_shape_list[0]

    model_path = Path(config.export_model_path)
    model_path.parent.mkdir(exist_ok=True, parents=True)
    extension = model_path.suffix

    if extension == ".onnx":
        with torch.no_grad():
            torch.onnx.export(model, input_tensor_list, model_path, input_names=input_names)
    elif extension == ".xml":
        import openvino as ov

        if config.export_to_ir_via_onnx:
            model_onnx_path = model_path.with_suffix(".onnx")
            with torch.no_grad():
                torch.onnx.export(model, input_tensor_list, model_onnx_path, input_names=input_names)
            ov_model = ov.convert_model(model_onnx_path)
        else:
            ov_model = ov.convert_model(model, example_input=input_tensor_list, input=tuple(input_shape_list))
            # Rename input nodes
            for input_node, input_name in zip(ov_model.inputs, input_names):
                input_node.node.set_friendly_name(input_name)
        ov.save_model(ov_model, model_path)
    else:
        raise ValueError(f"--export-model-path argument should have suffix `.xml` or `.onnx` but got {extension}")
    logger.info(f"Saved to {model_path}")
