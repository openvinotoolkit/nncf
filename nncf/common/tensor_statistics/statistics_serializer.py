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
import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nncf

METADATA_FILE = "statistics_metadata.json"


def sanitize_filename(filename: str) -> str:
    """
    Replaces any forbidden characters with an underscore.
    """
    return re.sub(r'[\/:*?"<>|]', "_", filename)


def load_metadata(dir_path: Path) -> Dict[str, Any]:
    """
    Loads the metadata, including the mapping and any other metadata information from the metadata file.
    :param dir_path: The directory where the metadata file is stored.
    :return: A dictionary containing the mapping and metadata.
    """
    metadata_file = dir_path / METADATA_FILE
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {"mapping": {}, "metadata": {}}


def save_metadata(metadata: Dict[str, Any], dir_path: Path) -> None:
    """
    Saves the mapping and metadata to the metadata file.
    :param metadata: The dictionary containing both the mapping and other metadata.
    :param dir_path: The directory where the metadata file will be stored.
    """
    metadata_file = dir_path / METADATA_FILE
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def load_from_dir(dir_path: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Loads statistics from gzip-compressed files in the given directory.
    :param dir_path: The path to the directory from which to load the statistics.
    :return: 1) A dictionary with the original statistic names as keys and the loaded statistics as values.
    2) Metadata dictionary.
    """
    statistics = {}
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise nncf.ValidationError("The provided directory path does not exist.")
    metadata = load_metadata(dir_path)
    mapping = metadata.get("mapping", {})

    for statistics_file in dir_path.iterdir():
        if statistics_file.name == METADATA_FILE:
            continue  # Skip the metadata file

        try:
            with gzip.open(statistics_file, "rb") as f:
                sanitized_name = statistics_file.name
                original_name = mapping.get(sanitized_name, sanitized_name)
                statistics[original_name] = pickle.load(f)
        except (pickle.UnpicklingError, IOError) as e:
            raise RuntimeError(f"Error loading statistics from {statistics_file.name}: {e}")
    return statistics, metadata.get("metadata", {})


def dump_to_dir(
    statistics: Dict[str, Any], dir_path: str, additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Dumps statistics to gzip-compressed files in the specified directory, while maintaining a mapping file.
    :param statistics: A dictionary with statistic names as keys and the statistic data as values.
    :param dir_path: The path to the directory where the statistics will be dumped.
    :param additional_metadata: A dictionary containing any additional metadata to be saved with the mapping.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    metadata, mapping = {}, {}

    for original_name, statistics_value in statistics.items():
        sanitized_name = sanitize_filename(original_name)
        file_path = dir_path / sanitized_name

        # Update the mapping
        mapping[sanitized_name] = original_name

        try:
            with gzip.open(file_path, "wb") as f:
                pickle.dump(statistics_value, f)
        except (IOError, pickle.PicklingError) as e:
            raise RuntimeError(f"Failed to write data to file {file_path}: {e}")

    # Add additional metadata if provided
    if additional_metadata:
        metadata["metadata"] = additional_metadata

    # Update the mapping in the metadata file
    metadata["mapping"] = mapping
    save_metadata(metadata, dir_path)
