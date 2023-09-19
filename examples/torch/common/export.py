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
from pathlib import Path

import openvino as ov
import torch
from openvino.tools.mo import convert_model

from examples.common.sample_config import SampleConfig
from examples.torch.common.example_logger import logger
from nncf.api.compression import CompressionAlgorithmController
from nncf.torch.exporter import generate_input_names_list


def export_model(ctrl: CompressionAlgorithmController, config: SampleConfig) -> None:
    """
    Export compressed model ot OpenVINO format.

    :param controller: The compression controller.
    :param config: Quantization config.
    """

    model = ctrl.model if config.no_strip_on_export else ctrl.strip()
    model = model.eval().cpu()

    input_names = generate_input_names_list(len(model.nncf.input_infos))
    input_tensor_list = []
    input_shape_list = []
    for info in model.nncf.input_infos:
        input_shape = tuple([1] + list(info.shape)[1:])
        input_tensor_list.append(torch.rand(input_shape))
        input_shape_list.append(input_shape)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    model_path = Path(output_dir) / f"{config.model_name}.xml"

    if not config.export_via_onnx:
        ov_model = convert_model(
            model,
            example_input=input_tensor_list,
            input_shape=input_shape_list,
        )
        # Rename input nodes
        for input_node, input_name in zip(ov_model.inputs, input_names):
            input_node.node.set_friendly_name(input_name)
        ov.serialize(ov_model, model_path)
    else:
        model_onnx_path = Path(output_dir) / f"{config.model_name}.onnx"
        with torch.no_grad():
            torch.onnx.export(model, tuple(input_tensor_list), model_onnx_path, input_names=input_names)
        ov_model = convert_model(model_onnx_path)
        ov.serialize(ov_model, model_path)

    logger.info(f"Saved to {model_path}")
