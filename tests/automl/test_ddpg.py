import pytest
import torch
import numpy as np
from nncf.automl.agent.ddpg.ddpg import DDPG

N_STATE = 5
N_ACTION = 2
N_HIDDEN_NODE = 3
SCALAR_ZERO = 0
SCALAR_ONE = 1

NUM_ACTIONS = [1, 5, 2.5, 0, -1, None]
@pytest.mark.parametrize('num_action', NUM_ACTIONS,
                         ids=['_'.join(['num_action', str(a)]) for a in NUM_ACTIONS])
def test_can_create_ddpg(num_action):

    if num_action is None or isinstance(num_action, float) or num_action <= 0:
        with pytest.raises((TypeError, ZeroDivisionError, RuntimeError)):
            DDPG(N_STATE, num_action, {})
    else:
        DDPG(N_STATE, num_action, {})


def test_random_action():
    ddpg = DDPG(N_STATE, N_ACTION)
    randomized_action = ddpg.random_action()
    assert randomized_action.shape[0] == N_ACTION
    # check action value within bound
    assert sum((randomized_action >= ddpg.LBOUND) & (randomized_action <= ddpg.RBOUND)) == N_ACTION


def test_select_action():
    ddpg = DDPG(N_STATE, N_ACTION, {"hidden1": N_HIDDEN_NODE, "hidden2": N_HIDDEN_NODE})

    ddpg.actor.load_state_dict(
        {state: torch.ones_like(param) for state, param in ddpg.actor.state_dict().items()}
    )

    s_t = [1.0] * N_STATE
    episode = 1000
    selected_action = ddpg.select_action(s_t, episode, decay_epsilon=False)
    assert all(selected_action == 1.0)

def test_update_policy():
    ddpg = DDPG(N_STATE, N_ACTION, 
        {
            "hidden1": N_HIDDEN_NODE, 
            "hidden2": N_HIDDEN_NODE,
        })

    ddpg.update_policy()
    assert ddpg.actor_target == STUB
    assert ddpg.critic_target == STUB
    assert ddpg.value_loss == STUB
    assert ddpg.policy_loss == STUB


TAU = 0.1
def create_two_ddpg_agents():
    source_ddpg = DDPG(N_STATE, N_ACTION, {"hidden1": N_HIDDEN_NODE, "hidden2": N_HIDDEN_NODE, "tau": TAU})
    source_ddpg.actor.load_state_dict(
        {state: torch.ones_like(param)/2 for state, param in source_ddpg.actor.state_dict().items()}
    )

    target_ddpg = DDPG(N_STATE, N_ACTION, {"hidden1": N_HIDDEN_NODE, "hidden2": N_HIDDEN_NODE, "tau": TAU})
    target_ddpg.actor.load_state_dict(
        {state: torch.ones_like(param) for state, param in target_ddpg.actor.state_dict().items()}
    )
    return source_ddpg, target_ddpg


def test_soft_update():
    source_ddpg, target_ddpg = create_two_ddpg_agents()
    target_ddpg.soft_update(target_ddpg.actor, source_ddpg.actor)

    for state_key, state_tensor in target_ddpg.actor.state_dict().items():
        assert torch.all(state_tensor == 0.95).item()


def test_hard_update():
    source_ddpg, target_ddpg = create_two_ddpg_agents()
    target_ddpg.hard_update(target_ddpg.actor, source_ddpg.actor)

    for state_key, state_tensor in target_ddpg.actor.state_dict().items():
        assert torch.all(state_tensor == 0.5).item()


REPLAY_MEMORY_SIZE = 5
def test_observe():
    def check_replay_buffer_size(length):
        assert ddpg.memory.limit == REPLAY_MEMORY_SIZE
        assert len(ddpg.memory.actions) == length
        assert len(ddpg.memory.rewards) == length
        assert len(ddpg.memory.terminals) == length
        assert len(ddpg.memory.observations) == length

    ddpg = DDPG(N_STATE, N_ACTION, {"rmsize": REPLAY_MEMORY_SIZE})
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
