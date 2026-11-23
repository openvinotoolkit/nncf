# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest

import nncf
from nncf.common.utils.os import is_windows
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.quantization.quantize_model import check_external_data_location
from tests.onnx.models import build_matmul_model


def _build_model_with_external_data(temp_dir: str):
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
    return model


def test_no_external_data():
    model = build_matmul_model()
    assert check_external_data_location(model, None) is None


def test_should_be_absolute():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        model = _build_model_with_external_data(temp_dir)

        invalid_path = "data_dir"
        msg = (
            f"BackendParameters.EXTERNAL_DATA_DIR should be an absolute path, but {invalid_path} was provided instead."
        )

        with pytest.raises(nncf.ValidationError, match=msg):
            check_external_data_location(model, invalid_path)


def test_not_accessible():
    if is_windows():
        pytest.skip("checked on linux only")

    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        model = _build_model_with_external_data(temp_dir)

        invalid_path = f"{temp_dir}_invalid"
        msg = re.escape(
            f"Data of TensorProto (tensor name: W) should be stored in {invalid_path}/model.data, "
            "but it doesn't exist or is not accessible."
        )

        with pytest.raises(nncf.ValidationError, match=msg):
            check_external_data_location(model, invalid_path)


def test_valid_external_data_dir():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        model = _build_model_with_external_data(temp_dir)
        assert check_external_data_location(model, temp_dir) == temp_dir


def test_compress_weights_ignores_external_data_dir_from_input_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    other_dir = tmp_path / "other"
    model_dir.mkdir()
    other_dir.mkdir()
    model = _build_model_with_external_data(str(model_dir))
    weight = onnx.numpy_helper.to_array(model.graph.initializer[0], str(model_dir))
    # A data file with the same name but different values in another directory.
    (weight + 100.0).tofile(other_dir / "model.data")
    # The input model already carries the NNCF metadata key that points to the other directory.
    entry = model.metadata_props.add()
    entry.key = MetadataKey.EXTERNAL_DATA_DIR.value
    entry.value = str(other_dir)

    monkeypatch.chdir(model_dir)
    compressed_model = nncf.compress_weights(model)

    x = np.ones((2, 3), dtype=np.float32)
    (actual,) = onnxruntime.InferenceSession(compressed_model.SerializeToString()).run(None, {"X": x})
    np.testing.assert_allclose(actual, x @ weight, atol=1e-2)
