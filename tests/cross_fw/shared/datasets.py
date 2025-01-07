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
from copy import deepcopy
from typing import List

import numpy as np


class MockDataset:
    def __init__(self, shape: List[int]):
        self.n = 0
        self.shape = deepcopy(shape)

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < 1:
            self.n += 1
            return np.ones(self.shape, dtype=np.float32)
        raise StopIteration
