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

from examples.common.sample_config import SampleConfig
from examples.tensorflow.common.logger import logger
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.exporter import generate_input_names_list


def export_model(compression_ctrl: PTCompressionAlgorithmController, config: SampleConfig) -> None:
    """
    Export compressed model. Supported only 'onnx' format.

    :param compression_ctrl: The controller of the compression algorithm.
    :param config: Config of examples.
    """
    save_path = config.to_onnx
    inference_model = compression_ctrl.prepare_for_inference()
    inference_model = inference_model.eval().cpu()
    input_names = generate_input_names_list(len(inference_model.input_infos))
    input_tensor_list = []
    for info in inference_model.input_infos:
        input_shape = tuple([1] + list(info.shape)[1:])
        input_tensor_list.append(torch.rand(input_shape))

    torch.onnx.export(inference_model, tuple(input_tensor_list), save_path, input_names=input_names)

    logger.info(f'Saved to {save_path}')
