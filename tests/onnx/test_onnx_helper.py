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

import tempfile

import onnx

from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import set_metadata
from nncf.onnx.graph.onnx_helper import get_array_from_tensor
from tests.onnx.models import build_matmul_model


def test_get_array_from_tensor():
    model = build_matmul_model()
    tensor = None
    for x in model.graph.initializer:
        if x.name == "W":
            tensor = x

    assert get_array_from_tensor(model, tensor).shape == (3, 2)


def test_get_array_from_tensor_external_data():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        model = build_matmul_model()
        model_path = f"{temp_dir}/model.onnx"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
            size_threshold=0,
            convert_attribute=False,
        )
        model = onnx.load_model(model_path, load_external_data=False)
        set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, temp_dir)

        tensor = None
        for x in model.graph.initializer:
            if x.name == "W":
                tensor = x
        assert get_array_from_tensor(model, tensor).shape == (3, 2)
