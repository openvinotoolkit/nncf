from nncf.automl.agent.ddpg.memory import RingBuffer

MAX_LEN = 2


def test_create_ring_buffer():
    RingBuffer(MAX_LEN)


def test_append():
    ring_buffer = RingBuffer(MAX_LEN)
    for i in range(MAX_LEN):
        ring_buffer.append(i)
        assert ring_buffer[i] == i
        assert len(ring_buffer) == i + 1

    assert ring_buffer.data == [0, 1]
    ring_buffer.append(3)
    assert ring_buffer.data == [3, 1]
