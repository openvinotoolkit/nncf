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
from pathlib import Path
from typing import Any, Dict

from nncf.common.utils.backend import BackendType


def validate_backend(metadata: Dict[str, Any], backend: BackendType) -> None:
    """
    Checks whether backend in metadata is equal to a provided backend.

    :param data: Loaded statistics.
    :param backend: Provided backend.
    """
    if "backend" not in metadata:
        raise ValueError("The provided metadata has no information about backend.")
    data_backend = metadata["backend"]
    if data_backend != backend.value:
        raise ValueError(
            f"Backend in loaded statistics {data_backend} does not match the expected backend {backend.value}."
        )


def validate_statistics_files_exist(metadata: Dict[str, Any], dir_path: Path) -> None:
    """
    Checks whether all statistics files exist.

    :param metadata: Loaded metadata.
    :param dir_path: Path to the cache directory.
    """
    for file_name in metadata["mapping"]:
        file_path = dir_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"One of the statistics file: {file_path} does not exist.")


def validate_cache(metadata: Dict[str, Any], dir_path: Path, backend: BackendType) -> None:
    """
    Validates cache directory.

    :param metadata: Metadata.
    :param dir_path: Path to the cache directory.
    :param backend: Backend type.
    """
    validate_backend(metadata, backend)
    validate_statistics_files_exist(metadata, dir_path)
