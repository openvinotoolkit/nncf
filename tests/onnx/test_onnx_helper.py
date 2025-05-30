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

import tempfile

import numpy as np
import onnx
import pytest
from numpy.testing import assert_array_equal

import nncf
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import set_metadata
from nncf.onnx.graph.onnx_helper import get_array_from_tensor
from nncf.onnx.graph.onnx_helper import pack_int4_to_uint8
from tests.onnx.models import build_matmul_model


def test_get_array_from_tensor():
    model = build_matmul_model()
    tensor = None
    for x in model.graph.initializer:
        if x.name == "W":
            tensor = x

    assert get_array_from_tensor(model, tensor).shape == (3, 2)


def test_get_array_from_tensor_external_data():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
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
        set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, temp_dir)

        tensor = None
        for x in model.graph.initializer:
            if x.name == "W":
                tensor = x
        assert get_array_from_tensor(model, tensor).shape == (3, 2)


def test_pack_int4_to_uint8_unsigned_even_block():
    weight = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)  # shape (4, 2)

    packed = pack_int4_to_uint8(weight, block_size=4, signed=False)
    expected = np.array(
        [
            [[(3 << 4) | 1, (7 << 4) | 5]],  # col 0: [1,3,5,7] -> [(3,1), (7,5)]
            [[(4 << 4) | 2, (8 << 4) | 6]],  # col 1: [2,4,6,8] -> [(4,2), (8,6)]
        ],
        dtype=np.uint8,
    )

    assert packed.shape == (2, 1, 2)
    assert_array_equal(packed, expected)


def test_pack_int4_to_uint8_unsigned_odd_block():
    weight = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.uint8)  # shape (5, 2)

    packed = pack_int4_to_uint8(weight, block_size=4, signed=False)
    expected = np.array(
        [
            [[(3 << 4) | 1, (7 << 4) | 5], [(0 << 4) | 9, 0]],
            [[(4 << 4) | 2, (8 << 4) | 6], [(0 << 4) | 10, 0]],
        ],
        dtype=np.uint8,
    )

    assert packed.shape == (2, 2, 2)
    assert_array_equal(packed, expected)


def test_pack_int4_to_uint8_signed():
    weight = np.array([[-8, -7], [-1, 0], [1, 2], [7, 7]], dtype=np.int8)

    # Apply offset: signed=True -> +8 to each value
    # Col 0: [-8, -1, 1, 7] -> [0, 7, 9, 15]
    # Packed as [(7 << 4) | 0, (15 << 4) | 9] = [112, 249]
    # Col 1: [-7, 0, 2, 7] -> [1, 8, 10, 15]
    # Packed as: [(8 << 4) | 1, (15 << 4) | 10] = [129, 250]

    packed = pack_int4_to_uint8(weight, block_size=4, signed=True)

    expected = np.array([[[(7 << 4) | 0, (15 << 4) | 9]], [[(8 << 4) | 1, (15 << 4) | 10]]], dtype=np.uint8)

    assert packed.shape == (2, 1, 2)
    assert_array_equal(packed, expected)


def test_pack_int4_to_uint8_multiple_blocks():
    weight = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.uint8)  # shape (8, 1)

    packed = pack_int4_to_uint8(weight, block_size=4, signed=False)
    expected = np.array(
        [
            [
                [(2 << 4) | 1, (4 << 4) | 3],  # block 0
                [(6 << 4) | 5, (8 << 4) | 7],  # block 1
            ]
        ],
        dtype=np.uint8,
    )

    assert packed.shape == (1, 2, 2)
    assert_array_equal(packed, expected)


def test_pack_int4_to_uint8_raises_on_invalid_signed_dtype():
    weight = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    with pytest.raises(nncf.ValidationError) as exc_info:
        pack_int4_to_uint8(weight, block_size=2, signed=True)

    assert "Expected weight dtype to be np.int8" in str(exc_info.value)
