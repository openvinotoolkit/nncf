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

import tempfile
from pathlib import Path

import openvino.runtime as ov
from openvino.tools import pot


def convert_openvino_model_to_compressed_model(
    model: ov.Model, target_device: str
) -> pot.graph.nx_model.CompressedModel:
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        xml_path = str(Path(tmp_dir) / "model.xml")
        bin_path = str(Path(tmp_dir) / "model.bin")
        ov.serialize(model, xml_path, bin_path)
        model_config = {
            "model_name": "model",
            "model": xml_path,
            "weights": bin_path,
        }
        pot_model = pot.load_model(model_config, target_device)

    return pot_model
