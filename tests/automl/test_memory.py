import pytest

from nncf.automl.agent.ddpg.memory import Memory, SequentialMemory, EpisodeParameterMemory, Experience, RingBuffer

WINDOW_LENGTH = 5
LIMIT = 5
MAX_LEN = 10

# TODO: temporary value to be replaced with actual one
STUB = RingBuffer(MAX_LEN)


def test_can_create_memory():
    Memory(WINDOW_LENGTH)
    SequentialMemory(LIMIT, window_length=WINDOW_LENGTH)
    EpisodeParameterMemory(LIMIT, window_length=WINDOW_LENGTH)


@pytest.mark.parametrize(
    ("memory", "ref_experiences"),
    [
        (SequentialMemory(LIMIT, window_length=WINDOW_LENGTH), Experience(STUB, STUB, STUB, STUB, STUB)),
        (EpisodeParameterMemory(LIMIT, window_length=WINDOW_LENGTH), Experience(STUB, STUB, STUB, STUB, STUB))
    ]
)
def test_sequential_sample(memory, ref_experiences):
    BATCH_SIZE = 10
    actual_experiences = memory.sample(BATCH_SIZE)
    assert actual_experiences == ref_experiences
