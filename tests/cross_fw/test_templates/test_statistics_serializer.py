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
from abc import abstractmethod
from pathlib import Path
from typing import Dict

import pytest

import nncf
from nncf.common.tensor_statistics.statistics_serializer import dump_statistics
from nncf.common.tensor_statistics.statistics_serializer import load_statistics
from nncf.common.utils.backend import BackendType
from nncf.common.utils.os import safe_open
from nncf.tensor.tensor import Tensor


class TemplateTestStatisticsSerializer:
    @abstractmethod
    def _get_backend_statistics(self) -> Dict[str, Dict[str, Tensor]]:
        """Returns a dictionary of statistics for testing purposes."""

    @abstractmethod
    def _get_backend(self) -> BackendType:
        """Returns the backend used for testing."""

    @abstractmethod
    def is_equal(self, a1: Dict[str, Tensor], a2: Dict[str, Tensor]) -> bool:
        """Determine if two statistics are equal."""

    def test_load_no_statistics_file(self, tmp_path):
        # Create a metadata file in the temp directory
        backend = self._get_backend()
        metadata = {"mapping": {"key1": "value1"}, "model": "test", "backend": backend.value}
        metadata_file = tmp_path / "statistics_metadata.json"
        with safe_open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Expect the load_statistics_from_dir to raise an error when trying to load non existed statistics
        with pytest.raises(nncf.StatisticsCacheError, match="One of the statistics file:"):
            load_statistics(tmp_path, backend)

    def test_dump_and_load_statistics(self, tmp_path):
        backend = self._get_backend()
        statistics = self._get_backend_statistics()
        additional_metadata = {"model": "facebook/opt-125m", "compression": "8-bit"}

        dump_statistics(statistics, tmp_path, backend, additional_metadata)

        assert len(list(Path(tmp_path).iterdir())) > 0, "No files created during dumping"

        metadata_file = tmp_path / "statistics_metadata.json"
        assert metadata_file.exists(), "Metadata file was not created"

        with safe_open(metadata_file, "r") as f:
            metadata = json.load(f)
            assert "mapping" in metadata, "Mapping is missing in metadata"
            assert metadata["model"] == "facebook/opt-125m"

        # Load the statistics and ensure it was loaded correctly
        loaded_statistics = load_statistics(tmp_path, backend)
        for layer_name, stat in statistics.items():
            assert layer_name in loaded_statistics, "Statistics not loaded correctly"
            assert self.is_equal(loaded_statistics[layer_name], stat)
