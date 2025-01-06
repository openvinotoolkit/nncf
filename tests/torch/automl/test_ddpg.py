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

import numpy as np
import pytest
import torch

from nncf.torch.automl.agent.ddpg.ddpg import DDPG

STUB = 0
N_STATE = 5
N_ACTION = 2
N_HIDDEN_NODE = 3
SCALAR_ZERO = 0
SCALAR_ONE = 1

VALID_NUM_ACTIONS = [1, 5]
VALID_NUM_STATES = [1, 3]


@pytest.mark.parametrize(
    "num_action", VALID_NUM_ACTIONS, ids=["_".join(["num_action", str(a)]) for a in VALID_NUM_ACTIONS]
)
@pytest.mark.parametrize("num_state", VALID_NUM_STATES, ids=["_".join(["num_state", str(s)]) for s in VALID_NUM_STATES])
def test_create_ddpg_with_valid_input(num_state, num_action):
    DDPG(num_state, num_action)


INVALID_VALUES = [2.5, 0, -1, None, "string"]


@pytest.mark.parametrize("num_action", INVALID_VALUES, ids=["_".join(["num_action", str(v)]) for v in INVALID_VALUES])
@pytest.mark.parametrize("num_state", INVALID_VALUES, ids=["_".join(["num_state", str(v)]) for v in INVALID_VALUES])
def test_create_ddpg_with_invalid_input(num_state, num_action):
    with pytest.raises((TypeError, ZeroDivisionError, RuntimeError)):
        DDPG(num_state, num_action, {})


def test_random_action():
    ddpg = DDPG(N_STATE, N_ACTION)
    randomized_action = ddpg.random_action()
    assert randomized_action.shape[0] == N_ACTION
    # check action value within bound
    assert sum((randomized_action >= ddpg.LBOUND) & (randomized_action <= ddpg.RBOUND)) == N_ACTION


EPISODE_NOISY_ACTION_TUPLES = [
    (0, [0.71018179, 0.82288581]),
    (10, [0.72084132, 0.82954315]),
    (100, [0.81406932, 0.88682551]),
    (1000, [0.99795472, 0.99875519]),
]


@pytest.mark.parametrize(
    "episode_action_pair",
    EPISODE_NOISY_ACTION_TUPLES,
    ids=["_".join(["episode", str(e)]) for e, _ in EPISODE_NOISY_ACTION_TUPLES],
)
@pytest.mark.parametrize(
    "decay_epsilon", [True, False], ids=["_".join(["actor_with_noise", str(b)]) for b in [True, False]]
)
def test_select_action(episode_action_pair, decay_epsilon, _seed):
    episode, reference_action = episode_action_pair

    hparams = {
        "hidden1": N_HIDDEN_NODE,
        "hidden2": N_HIDDEN_NODE,
        "init_delta": 0.5,
        "delta_decay": 0.99,
        "warmup_iter_number": 5,
    }

    ddpg = DDPG(N_STATE, N_ACTION, hparam_override=hparams)
    ddpg.actor.load_state_dict({state: torch.ones_like(param) for state, param in ddpg.actor.state_dict().items()})

    s_t = [1.0] * N_STATE
    selected_action = ddpg.select_action(s_t, episode, decay_epsilon)

    if decay_epsilon:
        np.testing.assert_allclose(selected_action, reference_action)
    else:
        assert all(selected_action == 1.0)


TEST_REFERENCES = [
    {"bs": 4, "discount": 1.0, "is_ma": False, "policy_loss": -0.3665157, "value_loss": 0.25},
    {"bs": 4, "discount": 0.9, "is_ma": True, "policy_loss": -0.3568782, "value_loss": 0.0013081},
    {"bs": 8, "discount": 1.0, "is_ma": True, "policy_loss": -0.3616864, "value_loss": 0},
    {"bs": 8, "discount": 0.8, "is_ma": False, "policy_loss": -0.3665157, "value_loss": 0.1828955},
]


@pytest.mark.parametrize(
    "test_vector",
    TEST_REFERENCES,
    ids=[
        "batch_size_{}-discount_factor_{}-moving_avg_{}".format(d["bs"], d["discount"], d["is_ma"])
        for d in TEST_REFERENCES
    ],
)
def test_update_policy(test_vector, mocker, _seed):
    batch_size, discount, is_movingavg, ref_policy_loss, ref_value_loss = test_vector.values()

    mocked_trace = mocker.patch("nncf.torch.automl.agent.ddpg.memory.SequentialMemory.sample_and_split")
    # state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
    mocked_trace.return_value = (
        np.ones((batch_size, N_STATE)),
        np.ones((batch_size, N_ACTION)),
        np.ones((batch_size, SCALAR_ONE)),
        np.ones((batch_size, N_STATE)),
        np.ones((batch_size, SCALAR_ONE)),
    )

    hparams = {
        "hidden1": N_HIDDEN_NODE,
        "hidden2": N_HIDDEN_NODE,
        "bsize": batch_size,
        "discount": discount,
        "window_length": SCALAR_ONE,
    }

    ddpg = DDPG(N_STATE, N_ACTION, hparam_override=hparams)
    ddpg.actor.load_state_dict({state: torch.ones_like(param) / 2 for state, param in ddpg.actor.state_dict().items()})
    ddpg.actor_target.load_state_dict(
        {state: torch.ones_like(param) for state, param in ddpg.actor_target.state_dict().items()}
    )

    ddpg.moving_average = is_movingavg
    ddpg.update_policy()
    np.testing.assert_almost_equal(ddpg.policy_loss.item(), ref_policy_loss)
    np.testing.assert_almost_equal(ddpg.value_loss.item(), ref_value_loss)


TAU = 0.1


def create_two_ddpg_agents():
    source_ddpg = DDPG(
        N_STATE, N_ACTION, hparam_override={"hidden1": N_HIDDEN_NODE, "hidden2": N_HIDDEN_NODE, "tau": TAU}
    )
    source_ddpg.actor.load_state_dict(
        {state: torch.ones_like(param) / 2 for state, param in source_ddpg.actor.state_dict().items()}
    )

    target_ddpg = DDPG(
        N_STATE, N_ACTION, hparam_override={"hidden1": N_HIDDEN_NODE, "hidden2": N_HIDDEN_NODE, "tau": TAU}
    )
    target_ddpg.actor.load_state_dict(
        {state: torch.ones_like(param) for state, param in target_ddpg.actor.state_dict().items()}
    )
    return source_ddpg, target_ddpg


def test_soft_update():
    source_ddpg, target_ddpg = create_two_ddpg_agents()
    target_ddpg.soft_update(target_ddpg.actor, source_ddpg.actor)

    for _, state_tensor in target_ddpg.actor.state_dict().items():
        assert torch.all(state_tensor == 0.95).item()


def test_hard_update():
    source_ddpg, target_ddpg = create_two_ddpg_agents()
    target_ddpg.hard_update(target_ddpg.actor, source_ddpg.actor)

    for _, state_tensor in target_ddpg.actor.state_dict().items():
        assert torch.all(state_tensor == 0.5).item()


REPLAY_MEMORY_SIZE = 5


def test_observe():
    def check_replay_buffer_size(length):
        assert ddpg.memory.limit == REPLAY_MEMORY_SIZE
        assert len(ddpg.memory.actions) == length
        assert len(ddpg.memory.rewards) == length
        assert len(ddpg.memory.terminals) == length
        assert len(ddpg.memory.observations) == length

    ddpg = DDPG(N_STATE, N_ACTION, hparam_override={"rmsize": REPLAY_MEMORY_SIZE})
    check_replay_buffer_size(SCALAR_ZERO)

    reward_t = SCALAR_ZERO
    states_t = [SCALAR_ZERO] * N_STATE
    action_t = [SCALAR_ZERO] * N_ACTION
    is_done_t = True

    ddpg.observe(reward_t, states_t, action_t, is_done_t)
    check_replay_buffer_size(SCALAR_ONE)

    # Replay Buffer Appending is disabled in non-training mode
    # size of experience must stay as the same as above
    ddpg.is_training = False
    ddpg.observe(reward_t, states_t, action_t, is_done_t)
    ddpg.observe(reward_t, states_t, action_t, is_done_t)
    check_replay_buffer_size(SCALAR_ONE)
