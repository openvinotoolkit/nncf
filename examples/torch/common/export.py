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

    if len(input_tensor_list) == 1:
        input_tensor_list = input_tensor_list[0]
        input_shape_list = input_shape_list[0]

    model_path = Path(config.to_ir)
    model_path.parent.mkdir(exist_ok=True, parents=True)

    if config.export_via_onnx:
        model_onnx_path = model_path.with_suffix(".onnx")
        with torch.no_grad():
            torch.onnx.export(model, input_tensor_list, model_onnx_path, input_names=input_names)
        ov_model = convert_model(model_onnx_path, compress_to_fp16=False)
    else:
        ov_model = convert_model(
            model, example_input=input_tensor_list, input_shape=input_shape_list, compress_to_fp16=False
        )
        # Rename input nodes
        for input_node, input_name in zip(ov_model.inputs, input_names):
            input_node.node.set_friendly_name(input_name)

    ov.save_model(ov_model, model_path)
    logger.info(f"Saved to {model_path}")
