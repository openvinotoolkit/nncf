"""
 Copyright (c) 2020 Intel Corporation
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
import queue
import numpy as np
import torch
from copy import deepcopy
from torch import optim

from nncf.utils import get_filters_num


class EvolutionOptimizer():
    def __init__(self, initial_filter_ranks, hparams):
        # Optimizer hyper-params
        self.POPULATIONS = hparams.get('POPULATIONS', 20)
        self.GENERATIONS = hparams.get('GENERATIONS', 400)
        self.SAMPLES = hparams.get('SAMPLES', 16)
        self.mutate_percent = hparams.get('mutate_percent', 0.1)
        self.SCALE_SIGMA = hparams.get('scale_sigma', 1)
        self.max_reward = hparams.get('max_reward', -np.inf)

        self.index_queue = queue.Queue(self.POPULATIONS)
        self.oldest_index = None
        self.population_reward = np.zeros(0)
        self.population_data = []
        self.mean_reward = []

        self.best_action = None
        self.num_layers = len(initial_filter_ranks)
        self.all_action_keys = np.array(list(initial_filter_ranks.keys()))
        self.original_dist_stat = {}
        for k in self.all_action_keys:
            a = initial_filter_ranks[k].cpu().detach().numpy()
            self.original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

        self.state = None
        self.reward = None
        self.episode = None
        self.info = None
        self.last_action = None

    def _save_episode_info(self, reward):
        print('Best reward ', self.max_reward)
        print('cur reward ', reward)
        if reward > self.max_reward:
            self.max_reward = reward
            self.best_action = self.last_action

        if self.episode < self.POPULATIONS:
            self.index_queue.put(self.episode)
            self.population_data.append(self.last_action)
            self.population_reward = np.append(self.population_reward, [reward])
        else:
            self.index_queue.put(self.oldest_index)
            self.population_data[self.oldest_index] = self.last_action
            self.population_reward[self.oldest_index] = reward

    def _predict_action(self):
        """
        Predict action for the last state.
        :return: action (pertrubation)
        """
        episode_num = self.episode
        step_size = 1 - (float(episode_num) / (self.GENERATIONS * 1.25))
        action = {}

        if episode_num == self.POPULATIONS - 1:
            for k in self.all_action_keys:
                action[k] = (1, 0)
        elif episode_num < self.POPULATIONS - 1:
            for k in self.all_action_keys:
                scale = np.exp(float(np.random.normal(0, self.SCALE_SIGMA)))
                shift = float(np.random.normal(0, self.original_dist_stat[k]['std']))
                action[k] = (scale, shift)
        else:
            self.mean_reward.append(np.mean(self.population_reward))
            sampled_idxs = np.random.choice(self.POPULATIONS, self.SAMPLES)
            sampled_rewards = self.population_reward[sampled_idxs]
            winner_idx_ = np.argmax(sampled_rewards)
            winner_idx = sampled_idxs[winner_idx_]

            # Mutate winner
            winner = self.population_data[winner_idx]
            # Perturbate winner
            mutate_num = int(self.mutate_percent * self.num_layers)
            mutate_candidate_idxs = np.random.choice(self.num_layers, mutate_num)
            mutate_candidates = self.all_action_keys[mutate_candidate_idxs]
            for k in self.all_action_keys:
                scale = 1
                shift = 0
                if k in mutate_candidates:
                    scale = np.exp(float(np.random.normal(0, self.SCALE_SIGMA * step_size)))
                    shift = float(np.random.normal(0, self.original_dist_stat[k]['std']))
                action[k] = (scale * winner[k][0], shift + winner[k][1])
            self.oldest_index = self.index_queue.get()
        return action

    def ask(self, episode_num):
        """
        Predict and returns action for the last told episode information: state, reward, episode_num and info
        :return:
        """
        self.episode = episode_num
        action = self._predict_action()
        self.last_action = action
        return action

    def tell(self, state, reward, end_of_episode, episode_num, info):
        """
        Getting info about episode step and save it every end of episode
        """
        # save state, reward and info from the current step
        self.state = state
        self.reward = reward
        self.episode = episode_num
        self.info = info

        if end_of_episode:
            self._save_episode_info(reward)


class LeGREvolutionEnv():
    def __init__(self, filter_pruner, model, train_loader, val_loader, train_fn, val_fn, config,
                 train_steps, pruning_max):
        self.loss_as_reward = True
        self.prune_target = pruning_max

        # Train/test params
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_fn = train_fn
        self.validate_fn = val_fn
        self.config = config

        self.filter_pruner = filter_pruner
        self.steps = train_steps

        self.model = model

    def reset(self):
        self.filter_pruner.reset()
        self.model.eval()

        self.full_flops = self.filter_pruner.get_flops_number_in_model()
        self.rest = self.full_flops
        self.last_act = (1, 0)

        return torch.zeros(1), [self.full_flops, self.rest]

    def train_steps(self, steps):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.train_fn(self.train_loader, self.model, optimizer, steps)

    def get_reward(self):
        return self.validate_fn(self.val_loader, self.model)

    def step(self, action):
        self.last_act = action
        new_state = torch.zeros(1)

        reduced = self.filter_pruner.prune(self.prune_target, action)
        print('train')
        self.train_steps(self.steps)

        print('test')
        acc, loss = self.get_reward()
        loss = loss.item()
        if self.loss_as_reward:
            reward = -loss
        else:
            reward = acc
        done = 1
        info = [self.full_flops, reduced]
        return new_state, reward, done, info


class LeGRPruner:
    def __init__(self, filter_pruner_ctrl, model):
        self.filter_pruner = filter_pruner_ctrl
        self.model = model
        self.model_params_copy = None
        self._save_model_weights()
        self.init_filter_ranks = {minfo.module_scope: self.filter_pruner.filter_importance(minfo.module.weight)
                                  for minfo in self.filter_pruner.pruned_module_groups_info.get_all_nodes()}

    def _save_model_weights(self):
        self.model_params_copy = deepcopy(self.model.state_dict())

    def _restore_model_weights(self):
        self.model.load_state_dict(self.model_params_copy)

    def reset_masks(self):
        for minfo in self.filter_pruner.pruned_module_groups_info.get_all_nodes():
            mask = self.filter_pruner._get_mask(minfo)
            mask = torch.ones(get_filters_num(minfo.module)).to(
                    minfo.module.weight.device)

    def reset(self):
        self._restore_model_weights()
        self.reset_masks()

    def get_flops_number_in_model(self):
        return self.filter_pruner.full_flops

    def prune(self, flops_pruning_target, ranking_coeffs):
        self.filter_pruner.ranking_coeffs = ranking_coeffs
        self.filter_pruner.set_pruning_rate(flops_pruning_target)
