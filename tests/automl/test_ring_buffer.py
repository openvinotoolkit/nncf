import pytest
from nncf.automl.agent.ddpg.memory import RingBuffer

MAX_LEN = 3


def test_create_ring_buffer():
    RingBuffer(MAX_LEN)


def test_append():
    def assert_elements(b, ref):
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
