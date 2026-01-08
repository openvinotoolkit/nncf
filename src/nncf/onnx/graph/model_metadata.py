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

from enum import Enum
from typing import Optional

import onnx


class MetadataKey(Enum):
    EXTERNAL_DATA_DIR = "nncf.metadata.external_data_dir"


def set_metadata(model: onnx.ModelProto, key: MetadataKey, value: str) -> None:
    """
    Sets a metadata property on an ONNX model.

    :param model: The ONNX model to which the metadata will be added.
    :param key: The metadata key.
    :param value: The string value to associate with the given metadata key.
    """
    entry = model.metadata_props.add()
    entry.key = key.value
    entry.value = value


def get_metadata(model: onnx.ModelProto, key: MetadataKey) -> Optional[str]:
    """
    Returns the metadata value associated with the given key from the ONNX model.

    :param model: The ONNX model.
    :param key: The key of the metadata value to retrieve.
    :return: The metadata value associated with the provided key, or None if the key is not found.
    """
    for prop in model.metadata_props:
        if prop.key == key.value:
            return prop.value
    return None


def remove_metadata(model: onnx.ModelProto, key: MetadataKey) -> None:
    """
    Removes a metadata property from an ONNX model.

    :param model: The ONNX model from which the metadata will be removed.
    :param key: The metadata key to be removed.
    """
    for prop in model.metadata_props:
        if prop.key == key.value:
            model.metadata_props.remove(prop)
            break
