"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Optional
from typing import List
from collections.abc import Sized
from nncf.data import Dataset


# TODO(andrey-churkin): The algorithms from the POT use the `__len__()` method.
# It should be removed when we change all algorithms.
class POTDataLoader(Sized):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._length = None

    def get_data(self, indices: Optional[List[int]] = None):
        return self._dataset.get_data(indices)

    def get_inference_data(self, indices: Optional[List[int]] = None):
        return self._dataset.get_inference_data(indices)

    def __len__(self) -> int:
        if self._length is None:
            self._length = POTDataLoader._get_length(self.get_data())
        return self._length

    @staticmethod
    def _get_length(iterable) -> int:
        length = 0
        for _ in iterable:
            length = length + 1

        return length
