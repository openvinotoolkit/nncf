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

# Modification Notes:
# This unit test is adapted from:
# https://github.com/keras-rl/keras-rl/blob/master/tests/rl/test_memory.py


import pytest

from nncf.torch.automl.agent.ddpg.memory import RingBuffer

MAX_LEN = 3


def assert_elements(b: RingBuffer, ref: list):
    assert len(b) == len(ref)
    for idx in range(b.maxlen):
        if idx >= len(ref):
            with pytest.raises(KeyError):
                _ = b[idx]
        else:
            assert b[idx] == ref[idx]


def create_ring_buffer(length):
    return RingBuffer(length)


def create_filled_ring_buffer(length):
    ring_buffer = create_ring_buffer(length)
    for i in range(length):
        ring_buffer.append(i)
    return ring_buffer


def test_create_ring_buffer():
    RingBuffer(MAX_LEN)


def test_append():
    ring_buffer = create_ring_buffer(MAX_LEN)
    for i in range(MAX_LEN):
        ring_buffer.append(i)
        assert ring_buffer[i] == i
        assert len(ring_buffer) == i + 1
    assert_elements(ring_buffer, [0, 1, 2])

    ring_buffer.append(3)
    assert_elements(ring_buffer, [1, 2, 3])
    ring_buffer.append(4)
    assert_elements(ring_buffer, [2, 3, 4])
    ring_buffer.append(5)
    assert_elements(ring_buffer, [3, 4, 5])


def test_delitem():
    ring_buffer = create_filled_ring_buffer(MAX_LEN)

    del ring_buffer[0]
    assert_elements(ring_buffer, [1, 2])

    del ring_buffer[-1]
    assert_elements(ring_buffer, [1])


def test_delslice():
    ring_buffer = create_filled_ring_buffer(10)

    del ring_buffer[0:0]
    assert_elements(ring_buffer, list(range(0, 10)))

    del ring_buffer[0:1]
    assert_elements(ring_buffer, list(range(1, 10)))

    del ring_buffer[::2]
    assert_elements(ring_buffer, list(range(1, 10))[1::2])

    del ring_buffer[1:4:2]
    assert_elements(ring_buffer, [2, 6])


def test_del_append():
    ring_buffer = create_filled_ring_buffer(10)

    del ring_buffer[7:]
    assert_elements(ring_buffer, list(range(0, 7)))

    ring_buffer.append(7)
    assert_elements(ring_buffer, list(range(0, 8)))

    del ring_buffer[::3]
    assert_elements(ring_buffer, [1, 2, 4, 5, 7])

    # Intentionally fill the buffer and continue appending
    for i in range(0, 20, 3):
        ring_buffer.append(i)
    assert_elements(ring_buffer, [4, 5, 7, 0, 3, 6, 9, 12, 15, 18])
