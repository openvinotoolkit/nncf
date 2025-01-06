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

import gc

import torch

from nncf.experimental.torch2.function_hook.weak_map import WeakUnhashableKeyMap


def test_set_get():
    map = WeakUnhashableKeyMap()
    obj = torch.tensor([1])
    map[obj] = 1
    assert map[obj] == 1


def test_on_del():
    map = WeakUnhashableKeyMap()
    obj1 = torch.tensor([1])
    obj2 = torch.tensor([2])
    map[obj1] = 1
    map[obj2] = 2
    assert len(map._data) == 2

    del obj1
    gc.collect()
    assert len(map._data) == 1
    assert map[obj2] == 2

    del obj2
    gc.collect()
    assert len(map._data) == 0


def test_set_same():
    map = WeakUnhashableKeyMap()
    obj1 = torch.tensor([1])
    map[obj1] = 1
    map[obj1] = 2
    assert len(map._data) == 1
    del obj1
    gc.collect()
    assert len(map._data) == 0
