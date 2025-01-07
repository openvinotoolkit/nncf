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
import queue
from copy import copy
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim

from nncf.config.config import NNCFConfig
from nncf.config.schemata.defaults import PRUNING_LEGR_GENERATIONS
from nncf.config.schemata.defaults import PRUNING_LEGR_MUTATE_PERCENT
from nncf.config.schemata.defaults import PRUNING_LEGR_NUM_SAMPLES
from nncf.config.schemata.defaults import PRUNING_LEGR_POPULATION_SIZE
from nncf.config.schemata.defaults import PRUNING_LEGR_SIGMA_SCALE
from nncf.torch.utils import get_filters_num


class EvolutionOptimizer:
    """
    Class for optimizing ranking coefficients for the model with evolution algorithm (agent).
    The evolution algorithm works as follows:
    1. For the first population_size steps it generates and returns random actions (generated with some prior
    information). For every action, it gets a reward (some measure whether this action is good or not). During these
    generations all action - reward pairs saving to the population.
    2. During remaining (generations - population_size) generations it predict action by next scheme:
        - Choosing random num_samples actions from population
        - Choosing the best one from sampled and mutate it
        - Return the resulting action
    During this generation's action - reward pairs saving by updating oldest actions in population.

    After all generations, the best action (with the best reward value) is returned.
    """

    def __init__(self, initial_filter_norms: Dict, hparams: Dict, random_seed: int):
        """
        :param initial_filter_norms: Initial filter norms needed to get std and var of filter norms in each leyer.
        :param hparams: hyperparams of the Optimizer, can contain population_size, num_generations, num_samples,
        mutate_percent, sigma_scale
        :param random_seed: random seed, thet should be set during action generation for reproducibility
        """
        self.random_seed = random_seed
        # Optimizer hyper-params
        self.population_size = hparams.get("population_size", PRUNING_LEGR_POPULATION_SIZE)
        self.num_generations = hparams.get("num_generations", PRUNING_LEGR_GENERATIONS)
        self.num_samples = hparams.get("num_samples", PRUNING_LEGR_NUM_SAMPLES)
        self.mutate_percent = hparams.get("mutate_percent", PRUNING_LEGR_MUTATE_PERCENT)
        self.scale_sigma = hparams.get("sigma_scale", PRUNING_LEGR_SIGMA_SCALE)
        self.max_reward = -np.inf
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
            self.initial_norms_stats[key] = {"mean": np.mean(layer_norms), "std": np.std(layer_norms)}

        self.cur_state = None
        self.cur_reward = None
        self.cur_episode = None
        self.cur_info = None

    def get_best_action(self):
        return self.best_action

    def _save_episode_info(self, reward: float) -> None:
        """
        Saving episode information: action-reward pairs and updating best_action/reward variables if needed.
        :param reward: reward for the current episode
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
        Predict action for the current episode. Works as described above.
        :return: new generated action
        """
        np.random.seed(self.random_seed)
        episode_num = self.cur_episode
        action = {}

        if episode_num < self.population_size - 1:
            # During first population_size generations, generates random actions
            for key in self.layer_keys:
                scale = np.exp(np.random.normal(0, self.scale_sigma))
                shift = np.random.normal(0, self.initial_norms_stats[key]["std"])
                action[key] = (scale, shift)
        elif episode_num == self.population_size - 1:
            # Adding identity action to population
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
                    shift = np.random.normal(0, self.initial_norms_stats[key]["std"])
                action[key] = (scale * best_action[key][0], shift + best_action[key][1])
            self.oldest_index = self.indexes_queue.get()
        return action

    def ask(self, episode_num: int) -> Dict:
        """
        Predict and returns action for the last told episode information: state, reward, episode_num and info
        :return: predicted action
        """
        self.cur_episode = episode_num
        action = self._predict_action()
        self.last_action = action
        return action

    def tell(self, state: torch.Tensor, reward: float, end_of_episode: bool, episode_num: int, info: List) -> None:
        """
        Getting info about episode step and save it every end of episode
        """
        # Saving state, reward and info from the current step
        self.cur_state = state
        self.cur_reward = reward
        self.cur_episode = episode_num
        self.cur_info = info

        if end_of_episode:
            self._save_episode_info(reward)


class LeGREvolutionEnv:
    """
    Environment class for optimizing the accuracy of the pruned model with different ranking coefficients.
    During 'step' environment doing step with received action calculates current reward and useful info and return it
    During 'reset' resetting Pruner and environment params changed during iteration.
    """

    def __init__(
        self,
        filter_pruner: "LeGRPruner",
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_fn: Callable,
        train_optimizer: Optional[torch.optim.Optimizer],
        val_fn: Callable,
        config: NNCFConfig,
        train_steps: int,
        pruning_max: float,
    ):
        """
        :param filter_pruner: LeGRPruner, should have an interface for pruning model and resetting pruner.
        :param model: target model for which ranking coefficients are trained
        :param train_loader: data loader for training the model
        :param val_loader: data loader for validating the model
        :param train_fn: callable for training the model
        :param train_optimizer: optional, optimizer for training the model
        :param val_fn: callable for validation of the model, returns acc, loss
        :param config: NNCF config for model compression
        :param train_steps: number of training steps to evaluate action (ranking coefficients set)
        :param pruning_max: pruning level for the model
        """
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
        """
        Resetting pruner params (all changes in the model made by training) and environment params changed during
        the step.
        :return: tuple with state and info : full flops in the model and number of flops that is rest in the model
        """
        self.filter_pruner.reset()
        self.model.eval()

        self.full_flops = self.filter_pruner.get_full_flops_number_in_model()
        self.rest = self.full_flops
        self.last_act = None
        return torch.zeros(1), [self.full_flops, self.rest]

    def _train_steps(self, steps: int) -> None:
        """
        Training model with train_fn for received steps number.
        :param steps: number of model training steps
        """
        optimizer = self.train_optimizer(self.model.parameters())
        self.train_fn(self.train_loader, self.model, optimizer, self.filter_pruner, steps)

    def _get_reward(self) -> Tuple[float, float, float]:
        """
        Validating model with validate_fn and return result in format: (acc, loss)
        """
        return self.validate_fn(self.model, self.val_loader)

    def step(self, action: Dict) -> Tuple[torch.Tensor, float, bool, List]:
        """
        1. Getting action (ranking coefficients)
        2. Making step with this action - prune model with ranking coefficients
        3. Getting a reward for this action- train model for some steps and validate it
        4. Returning new state (for current settings state is default and not used), reward,
         whether the episode is over or not (for current settings an episode is over after every step) and additional
          info (full flops in model and flops left in the model)
        :param action: ranking coefficients
        """
        self.last_act = action
        new_state = torch.zeros(1)

        reduced = self.filter_pruner.prune(self.prune_target, action)
        self._train_steps(self.steps)

        acc, loss = self._get_reward()
        if self.loss_as_reward:
            reward = -loss
        else:
            reward = acc
        done = 1
        info = [self.full_flops, reduced]
        return new_state, reward, done, info


class LeGRPruner:
    """
    Wrapper for pruning controller with a simplified interface, allowing prune model with received ranking coefficients
    and resetting all changes in the model made by the environment.
    """

    def __init__(self, filter_pruner_ctrl: "FilterPruningController", model: nn.Module):  # noqa: F821
        self.filter_pruner = filter_pruner_ctrl
        self.scheduler = copy(self.filter_pruner.scheduler)
        self.model = model
        self.model_params_copy = None
        self._save_model_weights()
        self.init_filter_norms = {
            node.node_name: self.filter_pruner.filter_importance(node.module.weight)
            for node in self.filter_pruner.pruned_module_groups_info.get_all_nodes()
        }

    def loss(self) -> float:
        """
        :return: loss for pruning algorithm
        """
        return self.filter_pruner.loss()

    def _save_model_weights(self) -> None:
        """
        Saving copy of all model parameters
        """
        self.model_params_copy = deepcopy(self.model.state_dict())

    def _restore_model_weights(self):
        """
        Restoring saved original model parameters to discard any changes in model weights.
        """
        self.model.load_state_dict(self.model_params_copy)

    def _reset_masks(self) -> None:
        """
        Resetting masks for all pruned nodes
        """
        for minfo in self.filter_pruner.pruned_module_groups_info.get_all_nodes():
            new_mask = torch.ones(get_filters_num(minfo.module)).to(minfo.module.weight.device)
            self.filter_pruner.set_mask(minfo, new_mask)

    def reset(self) -> None:
        """
        Resetting all changes made in the model (and model masks during environment step) by restoring the original
        model weights, resetting masks.
        """
        self._restore_model_weights()
        self._reset_masks()
        self.scheduler = copy(self.filter_pruner.scheduler)

    def get_full_flops_number_in_model(self) -> float:
        return self.filter_pruner.full_flops

    def prune(self, flops_pruning_target: float, ranking_coeffs: Dict) -> None:
        """
        Prune target model to flops pruning target with ranking_coeffs.
        :param flops_pruning_target: pruning target for the model pruning
        :param ranking_coeffs: ranking coefficients, that will be used for layers ranking during pruning
        """
        self.filter_pruner.ranking_coeffs = ranking_coeffs
        self.filter_pruner.set_pruning_level(flops_pruning_target)
