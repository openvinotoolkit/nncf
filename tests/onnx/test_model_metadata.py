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

import onnx

from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import get_metadata
from nncf.onnx.graph.model_metadata import remove_metadata
from nncf.onnx.graph.model_metadata import set_metadata
from tests.onnx.models import build_matmul_model


def test_set_and_get_metadata():
    model = build_matmul_model()
    path = "path/to/data/dir"
    set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, path)
    value = get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    assert value == path


def test_get_metadata_key_not_found():
    model = build_matmul_model()
    value = get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    assert value is None


def test_remove_metadata():
    model = build_matmul_model()
    path = "path/to/data/dir"
    set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, path)
    assert get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR) == path
    remove_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    assert get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR) is None


def test_remove_metadata_key_not_found():
    model = build_matmul_model()
    remove_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    assert get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR) is None


def _add_metadata_entry(model: onnx.ModelProto, key: str, value: str) -> None:
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def test_set_metadata_replaces_existing_value() -> None:
    model = build_matmul_model()
    # The input model may already carry entries with the same key.
    _add_metadata_entry(model, MetadataKey.EXTERNAL_DATA_DIR.value, "path/to/other/dir")
    _add_metadata_entry(model, MetadataKey.EXTERNAL_DATA_DIR.value, "path/to/another/dir")
    path = "path/to/data/dir"
    set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, path)
    assert get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR) == path
    assert [p.value for p in model.metadata_props if p.key == MetadataKey.EXTERNAL_DATA_DIR.value] == [path]


def test_remove_metadata_removes_all_entries() -> None:
    model = build_matmul_model()
    _add_metadata_entry(model, MetadataKey.EXTERNAL_DATA_DIR.value, "path/to/other/dir")
    _add_metadata_entry(model, "other.key", "value")
    _add_metadata_entry(model, MetadataKey.EXTERNAL_DATA_DIR.value, "path/to/data/dir")
    remove_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    assert get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR) is None
    assert [(p.key, p.value) for p in model.metadata_props] == [("other.key", "value")]
