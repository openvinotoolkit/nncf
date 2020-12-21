from nncf.automl.agent.ddpg.ddpg import DDPG

NUM_AGENT_STATES = 5
NB_ACTION = 1
# TODO: temporary value to be replaced with actual one
STUB = 0


def test_can_create_ddpg():
    DDPG(NUM_AGENT_STATES, NB_ACTION, {})


def test_update_policy():
    ddpg = DDPG(NUM_AGENT_STATES, NB_ACTION, {})
    ddpg.update_policy()
    assert ddpg.actor_target == STUB
    assert ddpg.critic_target == STUB
    assert ddpg.value_loss == STUB
    assert ddpg.policy_loss == STUB


def test_random_action():
    ddpg = DDPG(NUM_AGENT_STATES, NB_ACTION, {})
    actual_action = ddpg.random_action()
    assert actual_action == STUB


def test_select_action():
    ddpg = DDPG(NUM_AGENT_STATES, NB_ACTION, {})
    s_t = STUB
    episode = STUB
    selected_action = ddpg.select_action(s_t, episode)
    assert selected_action == STUB


def test_soft_update():
    ddpg = DDPG(NUM_AGENT_STATES, NB_ACTION, {})
    target = STUB
    ref_target = STUB
    source = STUB
    ddpg.soft_update(target, source)
    assert target == ref_target


def test_hard_update():
    ddpg = DDPG(NUM_AGENT_STATES, NB_ACTION, {})
    target = STUB
    ref_target = STUB
    source = STUB
    ddpg.hard_update(target, source)
    assert target == ref_target
