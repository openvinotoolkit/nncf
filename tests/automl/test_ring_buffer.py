"""
 Copyright (c) 2021 Intel Corporation
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

# Modification Notes:
# This unit test is adapted from:
# https://github.com/keras-rl/keras-rl/blob/master/tests/rl/test_memory.py


import pytest
from nncf.automl.agent.ddpg.memory import RingBuffer

MAX_LEN = 3


def test_create_ring_buffer():
    RingBuffer(MAX_LEN)


def test_append():
    def assert_elements(b: RingBuffer, ref: RingBuffer):
        assert len(b) == len(ref)
        for idx in range(b.maxlen):
            if idx >= len(ref):
                with pytest.raises(KeyError):
                    _ = b[idx]
            else:
                assert b[idx] == ref[idx]

    ring_buffer = RingBuffer(MAX_LEN)
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
