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
import time

from torch import nn

from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.pruning.filter_pruning.global_ranking.evolutionary_optimization import LeGRPruner, EvolutionOptimizer, \
    LeGREvolutionEnv
from nncf.torch.structures import LeGRInitArgs


class LeGR:
    """
    Class for training global ranking coefficients with Evolution optimization agent (but this agent can be easily
    replaced by any other RL agent with a similar interface) and LeGR-optimization environment.
    """
    def __init__(self, pruning_ctrl: 'FilterPruningController', target_model: nn.Module, legr_init_args: LeGRInitArgs,
                 train_steps: int = 200, generations: int = 400, max_pruning: float = 0.8, random_seed: int = 42):
        """
        Initializing all necessary structures for optimization- LeGREvolutionEnv environment and EvolutionOptimizer
         agent.
        :param pruning_ctrl: pruning controller, an instance of FilterPruningController class
        :param target_model: model for which layers ranking coefficient will be trained
        :param legr_init_args: initial arguments for LeGR algorithm
        :param train_steps: number of training steps to evaluate accuracy of some ranking coefficients (action of agent)
        :param generations: number of generations in evolution algorithm optimization
        :param max_pruning: pruning level of the model for which ranking coefficient will be optimized
        :param random_seed: random seed, that will be set during ranking coefficients generation
        """
        self.num_generations = generations
        self.max_pruning = max_pruning
        self.train_steps = train_steps

        self.pruner = LeGRPruner(pruning_ctrl, target_model)
        init_filter_norms = self.pruner.init_filter_norms
        agent_hparams = {
            'num_generations': self.num_generations
        }
        self.agent = EvolutionOptimizer(init_filter_norms, agent_hparams, random_seed)
        self.env = LeGREvolutionEnv(self.pruner, target_model, legr_init_args.train_loader,
                                    legr_init_args.val_loader, legr_init_args.train_steps_fn,
                                    legr_init_args.train_optimizer,
                                    legr_init_args.val_fn, legr_init_args.config,
                                    train_steps, max_pruning)

    def train_global_ranking(self):
        """
        Training of ranking coefficients. During every generation:
        1. Environment (LeGREvolutionEnv) send reward and useful info from the previous generation to the
        agent (EvolutionOptimizer)
        2. Agent generates new action (considering this information)
        3. Environment makes step with this action, calculates and return current reward from this
         action and some useful info

         In the end, an optimal action from the agent is returned.
        :return: optimal ranking coefficients (action)
        """
        reward_list = []

        nncf_logger.info('Start training LeGR ranking coefficients...')

        generation_time = 0
        end = time.time()
        for episode in range(self.num_generations):
            state, info = self.env.reset()

            # Beginning of the episode
            done = 0
            reward = 0
            episode_reward = []
            self.agent.tell(state, reward, done, episode, info)

            while not done:
                action = self.agent.ask(episode)
                new_state, reward, done, info = self.env.step(action)
                self.agent.tell(state, reward, done, episode, info)

                state = new_state
                episode_reward.append(reward)
            generation_time = time.time() - end
            end = time.time()

            nncf_logger.info('Generation = {episode}, '
                             'Reward = {reward:.3f}, '
                             'Time = {time:.3f} \n'.format(episode=episode, reward=episode_reward[0],
                                                           time=generation_time))
            reward_list.append(episode_reward[0])
        self.env.reset()
        nncf_logger.info('Finished training LeGR ranking coefficients.')
        nncf_logger.info('Evolution algorithm rewards history = {}'.format(reward_list))

        best_ranking = self.agent.get_best_action()
        return best_ranking
