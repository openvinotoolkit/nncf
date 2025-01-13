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
import json
from collections import defaultdict
from typing import Dict

import numpy as np
import pytest

import nncf
from nncf.common.tensor_statistics.statistics_serializer import add_unique_name
from nncf.common.tensor_statistics.statistics_serializer import load_metadata
from nncf.common.tensor_statistics.statistics_serializer import sanitize_filename
from nncf.common.tensor_statistics.statistics_serializer import save_metadata
from nncf.common.utils.backend import BackendType
from nncf.common.utils.os import safe_open
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend
from tests.cross_fw.test_templates.test_statistics_serializer import TemplateTestStatisticsSerializer


class TestNPStatisticsSerializer(TemplateTestStatisticsSerializer):
    def _get_backend_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "layer/1/activation": {"mean": Tensor(np.array([0.1, 0.2, 0.3]))},
            "layer/2/activation": {"variance": Tensor(np.array([0.05, 0.06, 0.07]))},
        }

    def _get_backend(self) -> TensorBackend:
        # any backend for numpy tensor, e.g. OpenVINO
        return BackendType.OPENVINO

    def is_equal(self, a1: Dict[str, Tensor], a2: Dict[str, Tensor]) -> bool:
        for key in a1:
            if key not in a2:
                return False
            if not np.array_equal(a1[key].data, a2[key].data):
                return False
        return True


def test_sanitize_filename():
    filename = "layer/1_mean/activation"
    sanitized = sanitize_filename(filename)
    assert sanitized == "layer_1_mean_activation", "Filename was not sanitized correctly"


def test_sanitize_filenames_with_collisions():
    filename_1 = "layer/1_mean:activation"
    filename_2 = "layer.1_mean/activation"
    unique_map = defaultdict(list)
    for filename in (filename_1, filename_2):
        sanitized = sanitize_filename(filename)
        add_unique_name(sanitized, unique_map)
    assert unique_map[sanitized] == ["layer_1_mean_activation_1", "layer_1_mean_activation_2"]


def test_load_metadata(tmp_path):
    # Create a metadata file in the temp directory
    metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
    metadata_file = tmp_path / "statistics_metadata.json"
    with safe_open(metadata_file, "w") as f:
        json.dump(metadata, f)

    loaded_metadata = load_metadata(tmp_path)
    assert loaded_metadata == metadata, "Metadata was not loaded correctly"


def test_load_no_existing_metadata(tmp_path):
    with pytest.raises(nncf.StatisticsCacheError, match="Metadata file does not exist in the following path"):
        load_metadata(tmp_path)


def test_save_metadata(tmp_path):
    metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
    save_metadata(metadata, tmp_path)

    metadata_file = tmp_path / "statistics_metadata.json"
    assert metadata_file.exists(), "Metadata file was not created"

    with safe_open(metadata_file, "r") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata == metadata, "Metadata was not saved correctly"
