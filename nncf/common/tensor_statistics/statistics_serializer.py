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
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

import nncf
from nncf.common.utils.backend import BackendType
from nncf.common.utils.os import safe_open
from nncf.tensor.definitions import TensorBackendType
from nncf.tensor.tensor import TTensor

METADATA_FILE = "statistics_metadata.json"


def sanitize_filename(filename: str) -> str:
    """
    Replaces forbidden characters in a filename with underscores.

    :param filename: Original filename.
    :return: Sanitized filename with no forbidden characters.
    """
    return re.sub(r'[\/:*?"<>|]', "_", filename)


def load_metadata(dir_path: Path) -> Dict[str, Any]:
    """
    Loads metadata from the specified directory.

    :param dir_path: Path to the directory containing the metadata file.
    :return: Dictionary containing metadata and mapping.
    """
    metadata_file = dir_path / METADATA_FILE
    if metadata_file.exists():
        with safe_open(metadata_file, "r") as f:
            return cast(Dict[str, Any], json.load(f))
    return {"mapping": {}, "metadata": {}}


def save_metadata(metadata: Dict[str, Any], dir_path: Path) -> None:
    """
    Saves metadata to a file in the specified directory.

    :param metadata: Dictionary containing metadata and mapping.
    :param dir_path: Path to the directory where the metadata file will be saved.
    """
    metadata_file = dir_path / METADATA_FILE
    with safe_open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def load_from_dir(dir_path: str, backend: TensorBackendType) -> Tuple[Dict[str, Dict[str, TTensor]], Dict[str, Any]]:
    """
    Loads statistics and metadata from a directory.

    :param dir_path: Path to the directory containing the data files.
    :param backend: Backend type to determine the loading function.
    :return: Tuple containing statistics and additional metadata.
    :raises nncf.ValidationError: If the directory does not exist.
    :raises nncf.InternalError: If an error occurs while loading a file.
    """
    statistics = {}
    path = Path(dir_path)
    if not path.exists():
        raise nncf.ValidationError("The provided directory path does not exist.")

    metadata = load_metadata(path)
    mapping = metadata.get("mapping", {})

    load_file_func = return_load_file_method(backend)
    for statistics_file in path.iterdir():
        if statistics_file.name == METADATA_FILE:
            continue  # Skip the metadata file

        try:
            sanitized_name = statistics_file.name
            original_name = mapping.get(sanitized_name, sanitized_name)
            statistics[original_name] = load_file_func(statistics_file)
        except Exception as e:
            raise nncf.InternalError(f"Error loading statistics from {statistics_file.name}: {e}")

    return statistics, metadata.get("metadata", {})


def dump_to_dir(
    statistics: Dict[str, Dict[str, TTensor]],
    dir_path: str,
    backend: TensorBackendType,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves statistics and metadata to a directory.

    :param statistics: Dictionary of statistics to save.
    :param dir_path: Path to the directory where files will be saved.
    :param backend: Backend type to determine the saving function.
    :param additional_metadata: Additional metadata to include in the metadata file.
    :raises RuntimeError: If an error occurs while saving a file.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

    metadata = {"mapping": {}}

    save_file_func = return_save_file_method(backend)
    for original_name, statistics_value in statistics.items():
        sanitized_name = sanitize_filename(original_name)
        file_path = path / sanitized_name

        metadata["mapping"][sanitized_name] = original_name

        try:
            save_file_func(statistics_value, file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to write data to file {file_path}: {e}")

    if additional_metadata:
        metadata["metadata"] = additional_metadata

    save_metadata(metadata, path)


def return_save_file_method(tensor_backend: TensorBackendType) -> Callable:
    """
    Returns the appropriate save_file function based on the backend.

    :param tensor_backend: Tensor backend type.
    :return: Function to save tensors.
    :raises RuntimeError: If the required module cannot be imported.
    """
    try:
        if tensor_backend == TensorBackendType.NUMPY:
            from safetensors.numpy import save_file

            return save_file
        if tensor_backend == TensorBackendType.TORCH:
            from safetensors.torch import save_file

            return save_file
    except ImportError as e:
        raise RuntimeError(f"Failed to import the required module: {e}")


def return_load_file_method(tensor_backend: TensorBackendType) -> Callable:
    """
    Returns the appropriate load_file function based on the backend.

    :param tensor_backend: Tensor backend type.
    :return: Function to load tensors.
    :raises RuntimeError: If the required module cannot be imported.
    """
    try:
        if tensor_backend == TensorBackendType.NUMPY:
            from safetensors.numpy import load_file

            return load_file
        if tensor_backend == TensorBackendType.TORCH:
            from safetensors.torch import load_file

            return load_file
    except ImportError as e:
        raise RuntimeError(f"Failed to import the required module: {e}")


def get_tensor_backend(backend: BackendType) -> TensorBackendType:
    """
    Maps a backend type to a tensor backend type.

    :param backend: Backend type.
    :return: Corresponding tensor backend type.
    :raises nncf.ValidationError: If the backend type is unsupported.
    """
    BACKEND_TO_TENSOR_BACKEND = {
        BackendType.OPENVINO: TensorBackendType.NUMPY,
        BackendType.ONNX: TensorBackendType.NUMPY,
        BackendType.TORCH_FX: TensorBackendType.TORCH,
        BackendType.TORCH: TensorBackendType.TORCH,
    }
    if backend not in BACKEND_TO_TENSOR_BACKEND:
        raise nncf.ValidationError(f"Unsupported backend type: {backend}")

    return BACKEND_TO_TENSOR_BACKEND[backend]
