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
import torch

from nncf.api.compression import CompressionAlgorithmController
from nncf.torch.exporter import count_tensors
from nncf.torch.exporter import generate_input_names_list
from nncf.torch.exporter import get_export_args


def export_model(ctrl: CompressionAlgorithmController, save_path: str, no_strip_on_export: bool) -> None:
    """
    Export compressed model. Supported only 'onnx' format.

    :param controller: The compression controller.
    :param save_path: Path to save onnx file.
    :param no_strip_on_export: Set to skip strip model before export.
    """

    model = ctrl.model if no_strip_on_export else ctrl.strip()

    model = model.eval().cpu()

    export_args = get_export_args(model, device="cpu")
    input_names = generate_input_names_list(count_tensors(export_args))

    with torch.no_grad():
        torch.onnx.export(model, export_args, save_path, input_names=input_names)
