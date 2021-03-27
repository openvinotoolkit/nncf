"""
 Copyright (c) 2021 Intel Corporation
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

from collections import OrderedDict
from typing import Dict, Tuple, List, Optional

import os

from nncf.debug import is_debug
from nncf.hw_config import HWConfigType
from nncf.common.utils.logger import logger
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.quantization.precision_init.base_init import BasePrecisionInitializer, BasePrecisionInitParams
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.structures import AutoQPrecisionInitArgs

from pathlib import Path
import os.path as osp
import time
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd
import re
from io import StringIO
from copy import deepcopy

from nncf.common.os import safe_open


class AutoQPrecisionInitParams(BasePrecisionInitParams):
    def __init__(self, user_init_args: AutoQPrecisionInitArgs,
                 dump_autoq_data: bool = False,
                 iter_number: int = 0,
                 compression_ratio: float = None,
                 eval_subset_ratio: float = None,
                 ddpg_hparams_dict: Dict = None,
                 hw_cfg_type: HWConfigType = None,
                 skip_constraint: bool = False,
                 finetune: bool = False,
                 bits: List[int] = None):
        super().__init__(user_init_args)
        self.dump_autoq_data = dump_autoq_data
        self.iter_number = iter_number
        self.compression_ratio = compression_ratio
        self.eval_subset_ratio = eval_subset_ratio
        if ddpg_hparams_dict is None:
            self.ddpg_hparams_dict = {}
        else:
            self.ddpg_hparams_dict = ddpg_hparams_dict
        self.hw_cfg_type = hw_cfg_type
        self.skip_constraint = skip_constraint
        self.finetune = finetune
        self.bits = bits

    @classmethod
    def from_config(cls, autoq_init_config_dict: Dict,
                    user_init_args: AutoQPrecisionInitArgs,
                    target_hw_config_type: Optional[HWConfigType]) -> 'AutoQPrecisionInitParams':
        dict_copy = autoq_init_config_dict.copy()
        dump_autoq_data = dict_copy.pop('dump_init_precision_data', False)
        iter_number = dict_copy.pop('iter_number', 0)
        compression_ratio = dict_copy.pop('compression_ratio', 0.15)
        eval_subset_ratio = dict_copy.pop('eval_subset_ratio', 1.0)
        skip_constraint = dict_copy.pop('skip_constraint', False)
        finetune = dict_copy.pop('finetune', False)
        bits = dict_copy.pop('bits',  [2, 4, 8])

        return cls(
            user_init_args=user_init_args,
            dump_autoq_data=dump_autoq_data,
            iter_number=iter_number,
            ddpg_hparams_dict=dict_copy,
            hw_cfg_type=target_hw_config_type,
            compression_ratio=compression_ratio,
            eval_subset_ratio=eval_subset_ratio,
            skip_constraint=skip_constraint,
            finetune=finetune,
            bits=bits)


class AutoQPrecisionInitializer(BasePrecisionInitializer):
    def __init__(self,
                 algo: 'ExperimentalQuantizationController',
                 params: AutoQPrecisionInitParams,
                 hw_precision_constraints: HardwareQuantizationConstraints):
        super().__init__(algo, params, hw_precision_constraints)
        self.quantization_controller = algo
        self._params = params
        self._init_args = params.user_init_args
        self._dump_autoq_data = params.dump_autoq_data
        self._iter_number = params.iter_number
        self._ddpg_hparams_override = params.ddpg_hparams_dict
        self._hw_cfg_type = params.hw_cfg_type

    def apply_init(self) -> SingleConfigQuantizerSetup:
        from nncf.automl.environment.quantization_env import QuantizationEnv
        from nncf.automl.agent.ddpg.ddpg import DDPG
        from nncf.debug import DEBUG_LOG_DIR

        if self._dump_autoq_data or is_debug():
            dump_dir = self._init_args.config.get('log_dir', None)
            if dump_dir is None:
                dump_dir = DEBUG_LOG_DIR
            self.dump_dir = Path(dump_dir) / Path("autoq") / Path("autoq_agent_dump")
            self.dump_dir.mkdir(parents=True, exist_ok=True)

            self.policy_dict = OrderedDict() #key: episode
            self.best_policy_dict = OrderedDict() #key: episode

            self._init_args.config['episodic_nncfcfg'] = str(self.dump_dir / "episodic_nncfcfg")
            os.makedirs(self._init_args.config['episodic_nncfcfg'], exist_ok=True)

            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.dump_dir)
                # log compression config to tensorboard
                self.tb_writer.add_text('AutoQ/run_config',
                                         json.dumps(self._init_args.config['compression'],
                                         indent=4, sort_keys=False).replace("\n", "\n\n"), 0)
            except ModuleNotFoundError:
                logger.warning("Tensorboard installation not found! Install tensorboard Python package "
                               "in order for AutoQ tensorboard statistics data to be dumped")

        start_ts = datetime.now()

        from nncf.automl.environment.quantization_env import QuantizationEnvParams
        env_params = QuantizationEnvParams(compression_ratio=self._params.compression_ratio,
            eval_subset_ratio=self._params.eval_subset_ratio,
            skip_constraint=self._params.skip_constraint,
            finetune=self._params.finetune,
            bits=self._params.bits,
            dump_init_precision_data=self._dump_autoq_data,
            log_dir=Path(DEBUG_LOG_DIR) / Path("autoq"))

        # Instantiate Quantization Environment
        env = QuantizationEnv(
            self._model,
            self.quantization_controller,
            self._hw_precision_constraints,
            self._init_args.data_loader,
            self._init_args.eval_fn,
            hw_config_type=self._hw_cfg_type,
            params=env_params)

        nb_state = len(env.state_list)
        nb_action = 1

        # Instantiate Automation Agent
        agent = DDPG(nb_state, nb_action, self._iter_number, hparam_override=self._ddpg_hparams_override)

        if self._dump_autoq_data and self.tb_writer is not None:
            # Need to replace '|' in nodestr (QuantizerId/QuantizerPointId)
            # to '+' as it is a special character in markdown 
            temp_df = deepcopy(env.master_df[env.state_list + ['n_op']])
            temp_df["modified_nodestr"] = list(map(lambda x: x.replace("|","+"), temp_df.index.tolist()))
            temp_df = temp_df.set_index("modified_nodestr").reset_index()
            self.tb_writer.add_text('AutoQ/state_embedding', temp_df.to_markdown())

        best_policy, best_reward = self._search(agent, env)

        end_ts = datetime.now()

        final_qid_vs_qconfig_map = env.select_config_for_actions(best_policy)

        final_quantizer_setup = self.quantization_controller.get_quantizer_setup_for_current_state()
        for qp_id, qconf in final_qid_vs_qconfig_map.items():
            final_quantizer_setup.quantization_points[qp_id].qconfig = qconf

        logger.info('[AutoQ] best_reward: {}'.format(best_reward))
        logger.info('[AutoQ] best_policy: {}'.format(best_policy))
        logger.info("[AutoQ] Search Complete")
        logger.info("[AutoQ] Elapsed time of AutoQ Precision Initialization (): {}".format(end_ts-start_ts))
        return final_quantizer_setup


    def _search(self, agent: 'DDPG', env: 'QuantizationEnv') -> Tuple[pd.Series, float]:
        best_reward = -math.inf
        episode = 0
        episode_reward = 0.
        observation = None
        transition_buffer = []  # Transition buffer

        while episode < self._iter_number:  # counting based on episode
            episode_start_ts = time.time()
            if observation is None:
                # reset if it is the start of episode
                env.reset()
                observation = deepcopy(env.get_normalized_obs(0))
                agent.reset(observation)

            if episode < agent.warmup_iter_number:
                action = agent.random_action()
            else:
                action = agent.select_action(observation, episode=episode)

            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = env.step(map_precision(action))
            observation2 = deepcopy(observation2)
            transition_buffer.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            # update
            episode_reward += reward
            observation = deepcopy(observation2)

            if done:  # end of episode
                logger.info(
                    '## Episode[{}], reward: {:.3f}, acc: {:.3f}, model_ratio: {:.3f}, model_size(MB): {:.2f}, BOP_ratio: {:.3f}\n' \
                    .format(episode, episode_reward, info['accuracy'], info['model_ratio'], info['model_size']/8e6, info['bop_ratio']))

                final_reward = transition_buffer[-1][0]

                for i, (_, s_t, _, a_t, done) in enumerate(transition_buffer):
                    # Revision of prev_action as it could be modified by constrainer -------
                    if i == 0:
                        prev_action = 0.0
                    else:
                        prev_action = env.master_df['action'][i-1] / 8 #ducktape scaling
                    if prev_action != s_t['prev_action']:
                        s_t['prev_action'] = prev_action
                    # EO ------------------------

                    agent.observe(final_reward, s_t, a_t, done)
                    if episode >= agent.warmup_iter_number:
                        for _ in range(agent.n_update):
                            agent.update_policy()

                agent.memory.append(
                    observation,
                    agent.select_action(observation, episode=episode),
                    0., False
                )

                # reset
                observation = None
                episode_reward = 0.
                transition_buffer = []

                value_loss = agent.get_value_loss()
                policy_loss = agent.get_policy_loss()
                delta = agent.get_delta()

                bit_stats_tt = env.qctrl.statistics()['Bitwidth distribution:']
                bit_stats_tt.set_max_width(100)
                bit_stats_df = pd.read_csv(
                    StringIO(re.sub(
                        r'[-+|=]', '', bit_stats_tt.draw())), sep=r'\s{2,}', engine='python').reset_index(drop=True)

                if final_reward > best_reward:
                    best_reward = final_reward
                    best_policy = deepcopy(env.master_df['action'])
                    # best_policy = env.master_df['action']
                    info_tuple = (episode, best_reward, info['accuracy'], info['model_ratio'], info['bop_ratio'])
                    self._dump_best_episode(info_tuple, bit_stats_df, env)
                    log_str = '## Episode[{}] New best policy: {}, reward: {:.3f}, acc: {:.3f}, model_ratio: {:.3f}, BOP_ratio: {:.3f}'\
                        .format(episode, best_policy.values.tolist(), best_reward,
                                info['accuracy'], info['model_ratio'],  info['bop_ratio'])
                    logger.info("\033[92m {}\033[00m" .format(log_str))

                episodic_info_tuple = (episode, final_reward, best_reward,
                                       info['accuracy'], info['model_ratio'], info['bop_ratio'],
                                       value_loss, policy_loss, delta)
                self._dump_episode(episodic_info_tuple, bit_stats_df, env, agent)

                episode_elapsed = time.time() - episode_start_ts

                logger.info('## Episode[{}] Policy: \n{}\n'.format(episode, env.master_df['action'].to_string()))
                logger.info('## Episode[{}] Elapsed: {:.3f}\n'.format(episode, episode_elapsed))

                episode += 1

        return best_policy, best_reward


    def _dump_best_episode(self, info_tuple: Tuple, bit_stats_df: pd.DataFrame, env: 'QuantizationEnv'):
        if self._dump_autoq_data:
            episode = info_tuple[0]
            self.best_policy_dict[episode] = env.master_df['action'].astype('int')

            pd.DataFrame(
                self.best_policy_dict.values(), index=self.best_policy_dict.keys()).T.sort_index(
                    axis=1, ascending=False).to_csv(
                        osp.join(self.dump_dir, "best_policy.csv"), index_label="nodestr")

            best_policy_string = self._generate_tensorboard_logging_string(
                bit_stats_df, env.master_df, info_tuple, env.skip_constraint)
            self.tb_writer.add_text('AutoQ/best_policy', best_policy_string, episode)


    def _dump_episode(self,
                      episodic_info_tuple: Tuple, bit_stats_df: pd.DataFrame,
                      env: 'QuantizationEnv', agent: 'DDPG'):
        if self._dump_autoq_data:
            episode, final_reward, _, accuracy, model_ratio, bop_ratio, _, _, _ = episodic_info_tuple

            # Save nncf compression cfg
            episode_cfgfile =  '{0}/{1:03d}_nncfcfg.json'.format(
                str(self._init_args.config['episodic_nncfcfg']), episode)

            with safe_open(Path(episode_cfgfile), "w") as outfile:
                json.dump(self._init_args.config, outfile, indent=4, sort_keys=False)

            self.policy_dict[episode] = env.master_df['action'].astype('int')
            pd.DataFrame(
                self.policy_dict.values(), index=self.policy_dict.keys()).T.sort_index(axis=1, ascending=False).to_csv(
                    osp.join(self.dump_dir, "policy_per_episode.csv"), index_label="nodestr")

            # log current episode policy and feedback as text
            info_tuple = (episode, final_reward, accuracy, model_ratio, bop_ratio)
            current_strategy_string = self._generate_tensorboard_logging_string(
                bit_stats_df, env.master_df, info_tuple, env.skip_constraint)
            self.tb_writer.add_text('AutoQ/current_policy', current_strategy_string, episode)

            # visualization over episode
            if self.tb_writer is not None:
                self._add_to_tensorboard(self.tb_writer, episodic_info_tuple)

            if episode % int((self._iter_number + 10) / 10) == 0:
                agent.save_model(self.dump_dir)


    def _add_to_tensorboard(self, tb_writer: 'SummaryWriter', log_tuple: Tuple):
        episode, final_reward, best_reward, \
            accuracy, model_ratio, bop_ratio, \
                value_loss, policy_loss, delta = log_tuple

        tb_writer.add_scalar('AutoQ/reward/last', final_reward, episode)
        tb_writer.add_scalar('AutoQ/reward/best', best_reward, episode)
        tb_writer.add_scalar('AutoQ/accuracy', accuracy, episode)
        tb_writer.add_scalar('AutoQ/model_ratio', model_ratio, episode)
        tb_writer.add_scalar('AutoQ/bop_ratio', bop_ratio, episode)
        tb_writer.add_scalar('AutoQ/agent/value_loss', value_loss, episode)
        tb_writer.add_scalar('AutoQ/agent/policy_loss', policy_loss, episode)
        tb_writer.add_scalar('AutoQ/agent/delta', delta, episode)


    def _generate_tensorboard_logging_string(self,
                                             bit_stats_df: pd.DataFrame, master_df: pd.DataFrame,
                                             info_tuple: Tuple, skip_constraint=False) -> str:
        qdf = master_df # For readibility
        episode, reward, accuracy, model_ratio, bop_ratio = info_tuple

        text_string = bit_stats_df.to_markdown() + "\n\n\n"
        text_string += "Episode: {:>4}, Reward: {:.3f}, ".format(episode, reward)
        text_string += "Accuracy: {:.3f}, Model_Size_Ratio: {:.3f}, BOP_Ratio: {:.3f}\n\n\n".format(accuracy, model_ratio, bop_ratio)

        for _, row_id in enumerate(qdf.index.tolist()):
            Qtype = '(WQ)' if qdf.is_wt_quantizer[row_id] else '(AQ)'

            if skip_constraint is False and \
                qdf.loc[row_id, 'action'] != qdf.loc[row_id, 'unconstrained_action']:

                text_string += "\t{} <= {} | {} {} \n".format(
                    str(int(qdf.loc[row_id, 'action'])), str(int(qdf.loc[row_id, 'unconstrained_action'])),
                    Qtype, row_id)

            else:
                text_string += "\t{} | {} {} \n".format(
                    str(int(qdf.loc[row_id, 'action'])), Qtype, row_id)

        return text_string

def map_precision(action: float) -> int:
    precision_set = [2, 4, 8]
    precision_set = np.array(sorted(precision_set))
    tuned_point = precision_set + 3
    max_bit = max(precision_set)

    i = None
    for i, point in enumerate(tuned_point):
        if action <= 2**point / 2**max_bit:
            return int(precision_set[i])
    return int(precision_set[i])
