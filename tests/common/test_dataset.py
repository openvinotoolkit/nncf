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
from nncf import Dataset


def test_dataset():
    raw_data = list(range(50))
    dataset = Dataset(raw_data)

    data_provider = dataset.get_data()
    retrieved_data_items = list(data_provider)
    assert all(raw_data[i] == retrieved_data_items[i] for i in range(len(raw_data)))


def test_dataset_with_transform_func():
    raw_data = list(range(50))
    dataset = Dataset(raw_data, transform_func=lambda it: 2 * it)

    data_provider = dataset.get_inference_data()
    retrieved_data_items = list(data_provider)
    assert all(2 * raw_data[i] == retrieved_data_items[i] for i in range(len(raw_data)))


def test_dataset_with_indices():
    raw_data = list(range(50))
    dataset = Dataset(raw_data)

    data_provider = dataset.get_data(indices=list(range(0, 50, 2)))
    retrieved_data_items = list(data_provider)
    assert all(raw_data[2 * i] == retrieved_data_items[i] for i in range(len(raw_data) // 2))


def test_dataset_with_transform_func_with_indices():
    raw_data = list(range(50))
    dataset = Dataset(raw_data, transform_func=lambda it: 2 * it)

    data_provider = dataset.get_inference_data(indices=list(range(0, 50, 2)))
    retrieved_data_items = list(data_provider)
    assert all(2 * raw_data[2 * i] == retrieved_data_items[i] for i in range(len(raw_data) // 2))


def test_dataset_without_length():
    raw_data = list(range(50))
    dataset_with_length = Dataset(raw_data)
    dataset_without_length = Dataset(iter(raw_data))
    assert dataset_with_length.get_length() == 50
    assert dataset_without_length.get_length() is None

    data_provider = dataset_with_length.get_data()
    retrieved_data_items = list(data_provider)
    assert all(raw_data[i] == retrieved_data_items[i] for i in range(len(raw_data)))

    data_provider = dataset_without_length.get_data()
    retrieved_data_items = list(data_provider)
    assert all(raw_data[i] == retrieved_data_items[i] for i in range(len(raw_data)))
