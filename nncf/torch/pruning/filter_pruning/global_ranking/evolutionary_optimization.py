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
from typing import List, Callable, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy, copy
from functools import partial
from torch import optim

from nncf.config.config import NNCFConfig
from nncf.torch.utils import get_filters_num


class EvolutionOptimizer:
    def __init__(self, initial_filter_norms: Dict, hparams: Dict, random_seed: int):
        self.random_seed = random_seed
        # Optimizer hyper-params
        self.population_size = hparams.get('population_size', 64)
        self.num_generations = hparams.get('num_generations', 400)
        self.num_samples = hparams.get('num_samples', 16)
        self.mutate_percent = hparams.get('mutate_percent', 0.1)
        self.scale_sigma = hparams.get('sigma_scale', 1)
        self.max_reward = hparams.get('max_reward', -np.inf)
        self.mean_rewards = []

        self.indexes_queue = queue.Queue(self.population_size)
        self.oldest_index = None
        self.population_rewards = np.zeros(self.population_size)
        self.population = [None for i in range(self.population_size)]

        self.best_action = None
        self.last_action = None
        self.num_layers = len(initial_filter_norms)
        self.layer_keys = np.array(list(initial_filter_norms.keys()))
        self.initial_norms_stats = {}
        for key in self.layer_keys:
            layer_norms = initial_filter_norms[key].cpu().detach().numpy()
            self.initial_norms_stats[key] = {'mean': np.mean(layer_norms), 'std': np.std(layer_norms)}

        self.cur_state = None
        self.cur_reward = None
        self.cur_episode = None
        self.cur_info = None

    def _save_episode_info(self, reward: float) -> None:
        """
        Savin episode information
        :param reward:
        :return:
        """
        # Update best action and reward if needed
        if reward > self.max_reward:
            self.best_action = self.last_action
            self.max_reward = reward

        if self.cur_episode < self.population_size:
            self.indexes_queue.put(self.cur_episode)
            self.population[self.cur_episode] = self.last_action
            self.population_rewards[self.cur_episode] = reward
        else:
            self.indexes_queue.put(self.oldest_index)
            self.population[self.oldest_index] = self.last_action
            self.population_rewards[self.oldest_index] = reward

    def _predict_action(self) -> Dict:
        """
        Predict action for the last state.
        :return: action
        """
        np.random.seed(self.random_seed)
        episode_num = self.cur_episode
        action = {}

        if episode_num < self.population_size - 1:
            for key in self.layer_keys:
                scale = np.exp(np.random.normal(0, self.scale_sigma))
                shift = np.random.normal(0, self.initial_norms_stats[key]['std'])
                action[key] = (scale, shift)
        elif episode_num == self.population_size - 1:
            # Adding identity transformation to population
            for key in self.layer_keys:
                action[key] = (1, 0)
        else:
            step_size = 1 - (float(episode_num) / (self.num_generations * 1.25))  # Rename this
            self.mean_rewards.append(np.mean(self.population_rewards))

            # 1. Sampling  num_samples actions from population and choosing the best one
            sampled_idxs = np.random.choice(len(self.population_rewards), self.num_samples)
            sampled_rewards = self.population_rewards[sampled_idxs]
            best_action = self.population[sampled_idxs[np.argmax(sampled_rewards)]]

            # 2. Mutate best action
            mutate_num = int(self.mutate_percent * self.num_layers)
            mutate_idxs = np.random.choice(self.layer_keys, mutate_num)
            for key in self.layer_keys:
                scale, shift = 1, 0
                if key in mutate_idxs:
                    scale = np.exp(np.random.normal(0, self.scale_sigma * step_size))
                    shift = np.random.normal(0, self.initial_norms_stats[key]['std'])
                action[key] = (scale * best_action[key][0], shift + best_action[key][1])
            self.oldest_index = self.indexes_queue.get()
        return action

    def ask(self, episode_num: int) -> Dict:
        """
        Predict and returns action for the last told episode information: state, reward, episode_num and info
        :return:
        """
        self.cur_episode = episode_num
        action = self._predict_action()
        self.last_action = action
        return action

    def tell(self, state: torch.Tensor, reward: float, end_of_episode: bool, episode_num: int, info: List) -> None:
        """
        Getting info about episode step and save it every end of episode
        """
        # save state, reward and info from the current step
        self.cur_state = state
        self.cur_reward = reward
        self.cur_episode = episode_num
        self.cur_info = info

        if end_of_episode:
            self._save_episode_info(reward)


class LeGREvolutionEnv:
    def __init__(self, filter_pruner: 'LeGRPruner', model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader, train_fn: Callable,
                 train_optimizer: Optional[torch.optim.Optimizer], val_fn: Callable, config: NNCFConfig,
                 train_steps: int, pruning_max: float):
        self.loss_as_reward = True
        self.prune_target = pruning_max
        self.steps = train_steps

        # Train/test params
        self.train_loader, self.val_loader = train_loader, val_loader
        self.train_fn = train_fn
        self.train_optimizer = train_optimizer
        if self.train_optimizer is None:
            # Default optimizer when the user did not provide a custom optimizer
            self.train_optimizer = partial(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.validate_fn = val_fn
        self.config = config

        self.filter_pruner = filter_pruner
        self.model = model

    def reset(self) -> Tuple[torch.Tensor, List]:
        self.filter_pruner.reset()
        self.model.eval()

        self.full_flops = self.filter_pruner.get_full_flops_number_in_model()
        self.rest = self.full_flops
        self.last_act = None
        return torch.zeros(1), [self.full_flops, self.rest]

    def _train_steps(self, steps: int) -> None:
        optimizer = self.train_optimizer(self.model.parameters())
        self.train_fn(self.train_loader, self.model, optimizer, self.filter_pruner, steps)

    def _get_reward(self) -> Tuple[float, float, float]:
        return self.validate_fn(self.model, self.val_loader)

    def step(self, action: Dict) -> Tuple[torch.Tensor, float, bool, List]:
        self.last_act = action
        new_state = torch.zeros(1)

        reduced = self.filter_pruner.prune(self.prune_target, action)
        self._train_steps(self.steps)

        acc, _, loss = self._get_reward()
        if self.loss_as_reward:
            reward = -loss
        else:
            reward = acc
        done = 1
        info = [self.full_flops, reduced]
        return new_state, reward, done, info


class LeGRPruner:
    def __init__(self, filter_pruner_ctrl: 'FilterPruningController', model: nn.Module):
        self.filter_pruner = filter_pruner_ctrl
        self.scheduler = copy(self.filter_pruner.scheduler)
        self.model = model
        self.model_params_copy = None
        self._save_model_weights()
        self.init_filter_norms = {node.node_name: self.filter_pruner.filter_importance(node.module.weight)
                                  for node in self.filter_pruner.pruned_module_groups_info.get_all_nodes()}

    def loss(self) -> float:
        return self.filter_pruner.loss()

    def _save_model_weights(self) -> None:
        self.model_params_copy = deepcopy(self.model.state_dict())

    def _restore_model_weights(self):
        self.model.load_state_dict(self.model_params_copy)

    def _reset_masks(self) -> None:
        for minfo in self.filter_pruner.pruned_module_groups_info.get_all_nodes():
            new_mask = torch.ones(get_filters_num(minfo.module)).to(
                minfo.module.weight.device)
            self.filter_pruner.set_mask(minfo, new_mask)

    def reset(self) -> None:
        self._restore_model_weights()
        self._reset_masks()
        self.scheduler = copy(self.filter_pruner.scheduler)

    def get_full_flops_number_in_model(self) -> float:
        return self.filter_pruner.full_flops

    def prune(self, flops_pruning_target: float, ranking_coeffs: Dict) -> None:
        self.filter_pruner.ranking_coeffs = ranking_coeffs
        self.filter_pruner.set_pruning_rate(flops_pruning_target)
