"""
 Copyright (c) 2023 Intel Corporation
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
import torch

from nncf.torch.exporter import generate_input_names_list
from nncf.torch.nncf_network import NNCFNetwork


def export_model(model: NNCFNetwork, save_path: str) -> None:
    """
    Export compressed model. Supported only 'onnx' format.

    :param model: The target model.
    :param save_path: Path to save onnx file.
    """
    model = model.eval().cpu()
    input_names = generate_input_names_list(len(model.input_infos))
    input_tensor_list = []
    for info in model.input_infos:
        input_shape = tuple([1] + list(info.shape)[1:])
        input_tensor_list.append(torch.rand(input_shape))

    with torch.no_grad():
        torch.onnx.export(model, tuple(input_tensor_list), save_path, input_names=input_names)
