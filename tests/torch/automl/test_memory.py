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
#
# The MIT License (MIT)
# Copyright (c) 2016 Matthias Plappert

# Modification Notes:
# This unit test is adapted from:
# https://github.com/keras-rl/keras-rl/blob/master/tests/rl/test_memory.py


import numpy as np
import pytest
from numpy.testing import assert_allclose

from nncf.torch.automl.agent.ddpg.memory import Memory
from nncf.torch.automl.agent.ddpg.memory import SequentialMemory

pytestmark = pytest.mark.legacy

WINDOW_LENGTH = 5
LIMIT = 5
MAX_LEN = 10


def test_can_create_memory():
    Memory(WINDOW_LENGTH)
    SequentialMemory(LIMIT, window_length=WINDOW_LENGTH)


def test_get_recent_state_with_episode_boundaries():
    memory = SequentialMemory(3, window_length=2, ignore_episode_boundaries=False)
    obs_size = (3, 4)

    obs0 = np.random.random(obs_size)

    obs1 = np.random.random(obs_size)
    terminal1 = False

    obs2 = np.random.random(obs_size)
    terminal2 = False

    obs3 = np.random.random(obs_size)
    terminal3 = True

    obs4 = np.random.random(obs_size)
    terminal4 = False

    obs5 = np.random.random(obs_size)
    terminal5 = True

    obs6 = np.random.random(obs_size)
    terminal6 = False

    state = np.array(memory.get_recent_state(obs0))
    assert state.shape == (2,) + obs_size
    assert np.allclose(state[0], 0.0)
    assert np.all(state[1] == obs0)

    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, 0, 0.0, terminal1)
    state = np.array(memory.get_recent_state(obs1))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs0)
    assert np.all(state[1] == obs1)

    memory.append(obs1, 0, 0.0, terminal2)
    state = np.array(memory.get_recent_state(obs2))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs1)
    assert np.all(state[1] == obs2)

    memory.append(obs2, 0, 0.0, terminal3)
    state = np.array(memory.get_recent_state(obs3))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs2)
    assert np.all(state[1] == obs3)

    memory.append(obs3, 0, 0.0, terminal4)
    state = np.array(memory.get_recent_state(obs4))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == np.zeros(obs_size))
    assert np.all(state[1] == obs4)

    memory.append(obs4, 0, 0.0, terminal5)
    state = np.array(memory.get_recent_state(obs5))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs4)
    assert np.all(state[1] == obs5)

    memory.append(obs5, 0, 0.0, terminal6)
    state = np.array(memory.get_recent_state(obs6))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == np.zeros(obs_size))
    assert np.all(state[1] == obs6)


def test_training_flag():
    obs_size = (3, 4)

    obs0 = np.random.random(obs_size)

    obs1 = np.random.random(obs_size)
    terminal1 = True

    obs2 = np.random.random(obs_size)
    terminal2 = False

    for training in (True, False):
        memory = SequentialMemory(3, window_length=2)

        state = np.array(memory.get_recent_state(obs0))
        assert state.shape == (2,) + obs_size
        assert np.allclose(state[0], 0.0)
        assert np.all(state[1] == obs0)
        assert memory.nb_entries == 0

        memory.append(obs0, 0, 0.0, terminal1, training=training)
        state = np.array(memory.get_recent_state(obs1))
        assert state.shape == (2,) + obs_size
        assert np.all(state[0] == obs0)
        assert np.all(state[1] == obs1)
        if training:
            assert memory.nb_entries == 1
        else:
            assert memory.nb_entries == 0

        memory.append(obs1, 0, 0.0, terminal2, training=training)
        state = np.array(memory.get_recent_state(obs2))
        assert state.shape == (2,) + obs_size
        assert np.allclose(state[0], 0.0)
        assert np.all(state[1] == obs2)
        if training:
            assert memory.nb_entries == 2
        else:
            assert memory.nb_entries == 0


def test_get_recent_state_without_episode_boundaries():
    memory = SequentialMemory(3, window_length=2, ignore_episode_boundaries=True)
    obs_size = (3, 4)

    obs0 = np.random.random(obs_size)

    obs1 = np.random.random(obs_size)
    terminal1 = False

    obs2 = np.random.random(obs_size)
    terminal2 = False

    obs3 = np.random.random(obs_size)
    terminal3 = True

    obs4 = np.random.random(obs_size)
    terminal4 = False

    obs5 = np.random.random(obs_size)
    terminal5 = True

    obs6 = np.random.random(obs_size)
    terminal6 = False

    state = np.array(memory.get_recent_state(obs0))
    assert state.shape == (2,) + obs_size
    assert np.allclose(state[0], 0.0)
    assert np.all(state[1] == obs0)

    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, 0, 0.0, terminal1)
    state = np.array(memory.get_recent_state(obs1))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs0)
    assert np.all(state[1] == obs1)

    memory.append(obs1, 0, 0.0, terminal2)
    state = np.array(memory.get_recent_state(obs2))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs1)
    assert np.all(state[1] == obs2)

    memory.append(obs2, 0, 0.0, terminal3)
    state = np.array(memory.get_recent_state(obs3))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs2)
    assert np.all(state[1] == obs3)

    memory.append(obs3, 0, 0.0, terminal4)
    state = np.array(memory.get_recent_state(obs4))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs3)
    assert np.all(state[1] == obs4)

    memory.append(obs4, 0, 0.0, terminal5)
    state = np.array(memory.get_recent_state(obs5))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs4)
    assert np.all(state[1] == obs5)

    memory.append(obs5, 0, 0.0, terminal6)
    state = np.array(memory.get_recent_state(obs6))
    assert state.shape == (2,) + obs_size
    assert np.all(state[0] == obs5)
    assert np.all(state[1] == obs6)


def test_sampling():
    memory = SequentialMemory(100, window_length=2, ignore_episode_boundaries=False)
    obs_size = (3, 4)
    actions = range(5)

    obs0 = np.random.random(obs_size)
    action0 = np.random.choice(actions)
    reward0 = np.random.random()

    obs1 = np.random.random(obs_size)
    terminal1 = False
    action1 = np.random.choice(actions)
    reward1 = np.random.random()

    obs2 = np.random.random(obs_size)
    terminal2 = False
    action2 = np.random.choice(actions)
    reward2 = np.random.random()

    obs3 = np.random.random(obs_size)
    terminal3 = True
    action3 = np.random.choice(actions)
    reward3 = np.random.random()

    obs4 = np.random.random(obs_size)
    terminal4 = False
    action4 = np.random.choice(actions)
    reward4 = np.random.random()

    obs5 = np.random.random(obs_size)
    terminal5 = False
    action5 = np.random.choice(actions)
    reward5 = np.random.random()

    terminal6 = False

    # memory.append takes the current observation, the reward after taking an action and if
    # the *new* observation is terminal, thus `obs0` and `terminal1` is correct.
    memory.append(obs0, action0, reward0, terminal1)
    memory.append(obs1, action1, reward1, terminal2)
    memory.append(obs2, action2, reward2, terminal3)
    memory.append(obs3, action3, reward3, terminal4)
    memory.append(obs4, action4, reward4, terminal5)
    memory.append(obs5, action5, reward5, terminal6)
    assert memory.nb_entries == 6

    experiences = memory.sample(batch_size=3, batch_idxs=[2, 3, 4])
    assert len(experiences) == 3

    assert_allclose(experiences[0].state0, np.array([obs1, obs2]))
    assert_allclose(experiences[0].state1, np.array([obs2, obs3]))
    assert experiences[0].action == action2
    assert experiences[0].reward == reward2
    assert experiences[0].terminal1 is True

    # Next experience has been re-sampled since since state0 would be terminal in which case we
    # cannot really have a meaningful transition because the environment gets reset. We thus
    # just ensure that state0 is not terminal.
    assert not np.all(experiences[1].state0 == np.array([obs2, obs3]))

    assert_allclose(experiences[2].state0, np.array([np.zeros(obs_size), obs4]))
    assert_allclose(experiences[2].state1, np.array([obs4, obs5]))
    assert experiences[2].action == action4
    assert experiences[2].reward == reward4
    assert experiences[2].terminal1 is False
