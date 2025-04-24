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

from typing import Any

import numpy as np
import onnxruntime as rt
from onnx import ModelProto

from nncf.common.engine import Engine
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import get_metadata


class ONNXEngine(Engine):
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, model: ModelProto, **rt_session_options: Any):
        self.input_names = set()

        sees_options = rt.SessionOptions()
        external_data_dir = get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
        if external_data_dir:
            sees_options.add_session_config_entry(
                "session.model_external_initializers_file_folder_path", external_data_dir
            )

        rt_session_options["providers"] = ["CPUExecutionProvider"]
        serialized_model = model.SerializeToString()
        self.sess = rt.InferenceSession(serialized_model, sees_options, **rt_session_options)

        for inp in self.sess.get_inputs():
            self.input_names.add(inp.name)

    def infer(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Runs model on the provided input via ONNXRuntime InferenceSession.
        Returns the dictionary of model outputs by node names.
        :param input_data: inputs for the model
        :return output_data: models outputs
        """
        output_tensors = self.sess.run([], input_data)
        model_outputs = self.sess.get_outputs()

        outputs_safe = {}
        for tensor, output in zip(output_tensors, model_outputs):
            # Workaround for https://github.com/microsoft/onnxruntime/issues/21922
            # After fixing this copying should be removed
            outputs_safe[output.name] = tensor.copy() if output.name in self.input_names else tensor

        return outputs_safe
