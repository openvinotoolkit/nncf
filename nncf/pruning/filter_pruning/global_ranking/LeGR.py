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
from nncf.pruning.filter_pruning.global_ranking.RL_evolution import EvolutionOptimizer, LeGREvolutionEnv, LeGRPruner
from nncf.nncf_logger import logger as nncf_logger


class LeGR:
    def __init__(self, pruning_ctrl, target_model, legr_init_args, train_steps=200, generations=400, max_pruning=0.8):
        self.GENERATIONS = generations

        self.pruner = LeGRPruner(pruning_ctrl, target_model)
        initial_filter_ranks = self.pruner.init_filter_ranks
        agent_hparams = {
            'generations': self.GENERATIONS
        }
        self.agent = EvolutionOptimizer(initial_filter_ranks, agent_hparams)
        self.env = LeGREvolutionEnv(self.pruner, target_model, legr_init_args.train_loader, legr_init_args.val_loader,
                                    legr_init_args.train_steps_fn, legr_init_args.val_fn, legr_init_args.config,
                                    train_steps, max_pruning)

    def train_global_ranking(self):
        reward_list = []

        for episode in range(self.GENERATIONS):
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

            nncf_logger.info('Generation = {}, reward = {}\n'.format(episode, episode_reward[0]))
            reward_list.append(episode_reward[0])
        self.env.reset()
        nncf_logger.info('Rewards history = {}'.format(reward_list))

        best_ranking = self.agent.best_action
        return best_ranking
