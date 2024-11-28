# Copyright (c) 2024 Intel Corporation
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
from abc import abstractmethod
from pathlib import Path
from typing import Dict

import pytest

import nncf
from nncf.common.tensor_statistics.statistics_serializer import dump_to_dir
from nncf.common.tensor_statistics.statistics_serializer import load_from_dir
from nncf.common.tensor_statistics.statistics_serializer import load_metadata
from nncf.common.tensor_statistics.statistics_serializer import sanitize_filename
from nncf.common.tensor_statistics.statistics_serializer import save_metadata
from nncf.tensor.definitions import TensorBackendType
from nncf.tensor.tensor import TTensor


class TemplateTestStatisticsSerializer:
    @abstractmethod
    def _get_backend_statistics(self) -> Dict[str, Dict[str, TTensor]]:
        """Returns a dictionary of statistics for testing purposes."""

    @abstractmethod
    def _get_tensor_backend(self) -> TensorBackendType:
        """Returns the backend used for testing."""

    @abstractmethod
    def is_equal(self) -> bool:
        """_summary_"""

    def test_sanitize_filename(self):
        filename = "layer/1_mean/activation"
        sanitized = sanitize_filename(filename)
        assert sanitized == "layer_1_mean_activation", "Filename was not sanitized correctly"

    def test_load_metadata(self, tmp_path):
        # Create a metadata file in the temp directory
        metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
        metadata_file = tmp_path / "statistics_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loaded_metadata = load_metadata(tmp_path)
        assert loaded_metadata == metadata, "Metadata was not loaded correctly"

    def test_save_metadata(self, tmp_path):
        metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
        save_metadata(metadata, tmp_path)

        metadata_file = tmp_path / "statistics_metadata.json"
        assert metadata_file.exists(), "Metadata file was not created"

        with open(metadata_file, "r") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata, "Metadata was not saved correctly"

    def test_dump_and_load_statistics(self, tmp_path):
        tensor_backend = self._get_tensor_backend()
        statistics = self._get_backend_statistics()
        additional_metadata = {"model": "facebook/opt-125m", "compression": "8-bit"}

        dump_to_dir(statistics, tmp_path, tensor_backend, additional_metadata)

        assert len(list(Path(tmp_path).iterdir())) > 0, "No files created during dumping"

        metadata_file = tmp_path / "statistics_metadata.json"
        assert metadata_file.exists(), "Metadata file was not created"

        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            assert "mapping" in metadata, "Mapping is missing in metadata"
            assert metadata["metadata"]["model"] == "facebook/opt-125m"

        # Load the statistics and ensure it was loaded correctly
        loaded_statistics, loaded_metadata = load_from_dir(tmp_path, tensor_backend)
        for layer_name, stat in statistics.items():
            assert layer_name in loaded_statistics, "Statistics not loaded correctly"
            assert self.is_equal(loaded_statistics[layer_name], stat)
        assert loaded_metadata["model"] == "facebook/opt-125m", "Metadata not loaded correctly"

    @pytest.mark.parametrize("tensor_backend", list(TensorBackendType))
    def test_invalid_statistics_file(self, tmp_path, tensor_backend):
        # Create a corrupt gzip file in the directory
        invalid_file = tmp_path / "invalid_file"
        with open(invalid_file, "w") as f:
            f.write("This is not a valid file")

        # Expect the load_from_dir to raise an error when trying to load the invalid file
        with pytest.raises(nncf.InternalError, match="Error loading statistics"):
            load_from_dir(tmp_path, tensor_backend)
