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
from pathlib import Path

import pytest

from nncf.common.tensor_statistics.statistics_serializer import dump_to_dir
from nncf.common.tensor_statistics.statistics_serializer import load_from_dir
from nncf.common.tensor_statistics.statistics_serializer import load_metadata
from nncf.common.tensor_statistics.statistics_serializer import sanitize_filename
from nncf.common.tensor_statistics.statistics_serializer import save_metadata


def test_sanitize_filename():
    filename = "layer/1_mean/activation"
    sanitized = sanitize_filename(filename)
    assert sanitized == "layer_1_mean_activation", "Filename was not sanitized correctly"


def test_load_metadata(tmp_path):
    # Create a metadata file in the temp directory
    metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
    metadata_file = tmp_path / "statistics_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    loaded_metadata = load_metadata(tmp_path)
    assert loaded_metadata == metadata, "Metadata was not loaded correctly"


def test_save_metadata(tmp_path):
    metadata = {"mapping": {"key1": "value1"}, "metadata": {"model": "test"}}
    save_metadata(metadata, tmp_path)

    metadata_file = tmp_path / "statistics_metadata.json"
    assert metadata_file.exists(), "Metadata file was not created"

    with open(metadata_file, "r") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata == metadata, "Metadata was not saved correctly"


def test_dump_and_load_statistics(tmp_path):
    statistics = {"layer/1_mean/activation": [0.1, 0.2, 0.3], "layer/2_variance": [0.05, 0.06, 0.07]}
    additional_metadata = {"model": "facebook/opt-125m", "compression": "8-bit"}

    dump_to_dir(statistics, tmp_path, additional_metadata)

    assert len(list(Path(tmp_path).iterdir())) > 0, "No files created during dumping"

    metadata_file = tmp_path / "statistics_metadata.json"
    assert metadata_file.exists(), "Metadata file was not created"

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        assert "mapping" in metadata, "Mapping is missing in metadata"
        assert metadata["metadata"]["model"] == "facebook/opt-125m"

    # Load the statistics and ensure it was loaded correctly
    loaded_statistics, loaded_metadata = load_from_dir(tmp_path)
    assert "layer/1_mean/activation" in loaded_statistics, "Statistics not loaded correctly"
    assert loaded_statistics["layer/1_mean/activation"] == [0.1, 0.2, 0.3]
    assert loaded_metadata["model"] == "facebook/opt-125m", "Metadata not loaded correctly"


def test_invalid_gzip_file(tmp_path):
    # Create a corrupt gzip file in the directory
    invalid_file = tmp_path / "invalid_file.gz"
    with open(invalid_file, "w") as f:
        f.write("This is not a valid gzip file")

    # Expect the load_from_dir to raise an error when trying to load the invalid file
    with pytest.raises(RuntimeError, match="Error loading statistics"):
        load_from_dir(tmp_path)
