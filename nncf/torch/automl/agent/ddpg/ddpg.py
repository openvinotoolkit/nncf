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


from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from nncf.torch.automl.agent.ddpg.memory import SequentialMemory

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats

    return stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=size)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super().__init__()
        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = self.fc11(x) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG:
    LBOUND = 0.0
    RBOUND = 1.0

    def __init__(self, nb_states, nb_actions, iter_number: int = None, hparam_override: dict = None):
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        hyperparameters = {
            "seed": 528,
            "train_episode": 600,
            "delta_decay": 0.995,
            "hidden1": 300,
            "hidden2": 300,
            "init_w": 0.003,
            "lr_c": 0.001,
            "lr_a": 0.0001,
            "rmsize": 2048,
            "window_length": 1,
            "bsize": 64,
            "tau": 0.01,
            "discount": 1.0,
            "epsilon": 50000,
            "init_delta": 0.5,
            "warmup_iter_number": 20,
            "n_update": 1,
        }

        if hparam_override is not None:
            for hparam, hparam_val in hparam_override.items():
                if hparam in hyperparameters:
                    hyperparameters[hparam] = hparam_val
        if iter_number is not None:
            hyperparameters["train_episode"] = iter_number

        args = SimpleNamespace(**hyperparameters)

        if args.seed is None:
            self.seed(0)
        else:
            self.seed(args.seed)

        # Create Actor and Critic Network
        net_cfg = {"hidden1": args.hidden1, "hidden2": args.hidden2, "init_w": args.init_w}

        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.n_update = args.n_update
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.update_delta_decay_factor(args.train_episode)
        self.warmup_iter_number = args.warmup_iter_number
        self.delta = args.init_delta

        # loss
        self.value_loss = 0.0
        self.policy_loss = 0.0

        self.epsilon = 1.0
        self.is_training = True

        if USE_CUDA:
            self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_delta_decay_factor(self, num_train_episode):
        assert num_train_episode > 0, "Number of train episode must be larger than 0"

        if num_train_episode < 1000:
            # every hundred of episode below 1000 (100, 200 ... 1000)
            calibrated_factor = [0.960, 0.960, 0.980, 0.987, 0.992, 0.993, 0.995, 0.995, 0.996, 0.996]
            self.delta_decay = calibrated_factor[num_train_episode // 100]

        elif num_train_episode <= 3000:
            self.delta_decay = (num_train_episode - 999) * (0.999 - 0.997) / (3000 - 999) + 0.997

        else:
            self.delta_decay = 0.999

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(
            self.batch_size
        )

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(
                [
                    to_tensor(next_state_batch),
                    self.actor_target(to_tensor(next_state_batch)),
                ]
            )

        target_q_batch = (
            to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values
        )

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = (-1) * self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        # update for log
        self.value_loss = value_loss
        self.policy_loss = policy_loss

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)  # save to memory

    def random_action(self):
        action = np.random.uniform(self.LBOUND, self.RBOUND, self.nb_actions)
        return action

    def select_action(self, s_t, episode, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)

        if decay_epsilon is True:
            self.delta = self.init_delta * (self.delta_decay ** (episode - self.warmup_iter_number))
            action = sample_from_truncated_normal_distribution(
                lower=self.LBOUND, upper=self.RBOUND, mu=action, sigma=self.delta, size=self.nb_actions
            )

        return np.clip(action, self.LBOUND, self.RBOUND)

    def reset(self, obs):
        pass

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))

        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_delta(self):
        return self.delta

    def get_value_loss(self):
        return self.value_loss

    def get_policy_loss(self):
        return self.policy_loss
