from collections import OrderedDict
from typing import List, Dict

import os
from torch import Tensor, nn

from nncf.nncf_logger import logger
from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.quantizer_id import QuantizerId
from nncf.structures import AutoQPrecisionInitArgs

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
from types import SimpleNamespace
from tensorboardX import SummaryWriter

class AutoQPrecisionInitializer:
    def __init__(self, algo: 'QuantizationController', init_precision_config,
                 init_args: AutoQPrecisionInitArgs):
        self.quantization_controller = algo
        self.init_args = init_args


    def apply_init(self):
        from nncf.automl.environment.quantization_env import QuantizationEnv
        from nncf.automl.agent.ddpg.ddpg import DDPG

        start_ts = datetime.now()

        # Instantiate Quantization Environment
        env = QuantizationEnv(
            self.quantization_controller,
            self.init_args.data_loader,
            self.init_args.eval_fn,
            self.init_args.config)

        nb_state = len(env.state_list)
        nb_action = 1

        # Instantiate Automation Agent
        agent = DDPG(nb_state, nb_action,
                    hparam_override=self.init_args.config['compression']['initializer']['precision'])

        best_policy, best_reward = self._search(agent, env, self.init_args.config)

        end_ts = datetime.now()

        self.set_chosen_config(dict(zip(env.master_df.qid_obj, best_policy)))

        logger.info('[AutoQ] best_reward: {}'.format(best_reward))
        logger.info('[AutoQ] best_policy: {}'.format(best_policy))
        logger.info("[AutoQ] Search Complete")
        logger.info("[AutoQ] Elapsed time of AutoQ Precision Initialization (): {}".format(end_ts-start_ts))


    def set_chosen_config(self, qid_bw_map: Dict[QuantizerId, int]):
        for qid, bw in qid_bw_map.items():
            self.quantization_controller.all_quantizations[qid].num_bits = bw


    def _search(self, agent, env, config):
        # def map_precision(action):
        #     action = float(action)
        #     min_bit, max_bit = (1,3)
        #     action = (max_bit - min_bit) * action + min_bit
        #     action = int(np.round(action, 0))
        #     action = 2**action
        #     return int(action)

        def map_precision(action):
            precision_set = [2,4,8]
            precision_set = np.array(sorted(precision_set))
            tuned_point = precision_set+3
            max_bit = max(precision_set)

            for i, point in enumerate(tuned_point):
                if action <= 2**point/2**max_bit:
                    return int(precision_set[i])
            return int(precision_set[i])

        # def map_precision(action):
        #     action = float(action)
        #     min_exp, max_exp = (1,3)
        #     lbound, rbound = min_exp - 0.5, max_exp + 0.5
        #     action = (rbound - lbound) * action + lbound
        #     action = np.round(action, 0)
        #     action = np.clip(action, min_exp, max_exp)
        #     action = int(2**action)
        #     return action

        assert 'autoq' == config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {})
        autoq_cfg = config.get('compression', {}).get('initializer', {}).get('precision')
        config['episodic_nncfcfg'] = osp.join(config['log_dir'], "episodic_nncfcfg")
        os.makedirs(config['episodic_nncfcfg'], exist_ok=True)

        args = SimpleNamespace(**autoq_cfg)

        policy_dict=OrderedDict() #key: episode
        best_policy_dict=OrderedDict() #key: episode

        num_episode = args.iter_number

        # best record
        best_reward = -math.inf
        best_policy = []

        tfwriter = SummaryWriter(config['log_dir'])

        log_cfg=OrderedDict()
        log_cfg['compression']=config['compression']
        tfwriter.add_text('AutoQ/run_config', json.dumps(log_cfg, indent=4, sort_keys=False).replace("\n", "\n\n"), 0)
        tfwriter.add_text('AutoQ/state_embedding', env.master_df[env.state_list].to_markdown())

        agent.is_training = True
        step = episode = episode_steps = 0
        episode_reward = 0.
        observation = None
        T = []  # Transition buffer

        while episode < num_episode:  # counting based on episode
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
            T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

            # [optional] save intermideate model
            if episode % int((num_episode+10)/10) == 0:
                agent.save_model(config['log_dir'])

            # update
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done:  # end of episode
                logger.info(
                    '#{}: episode_reward:{:.3f} acc: {:.3f}, model_ratio: {:.3f}, model_size(MB): {:.2f}\n'.format(
                        episode, episode_reward, info['accuracy'], info['model_ratio'], info['model_size']/8e6))

                final_reward = T[-1][0]

                for i, (_, s_t, s_t1, a_t, done) in enumerate(T):
                    # Revision of prev_action as it could be modified by constrainer -------
                    if i == 0:
                        prev_action = 0.0
                    else:
                        prev_action = env.master_df['action'][i-1] / 8 #ducktape scaling
                    if prev_action != s_t['prev_action']:
                        # print(i, prev_action, " <= ", s_t['prev_action'])
                        s_t['prev_action'] = prev_action
                    # EO ------------------------

                    agent.observe(final_reward, s_t, s_t1, a_t, done)
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
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                T = []

                # Save nncf compression cfg
                episode_cfgfile = osp.join(env.config['episodic_nncfcfg'], '{0:03d}_nncfcfg.json'.format(episode))
                with open(episode_cfgfile, "w") as outfile:
                    json.dump(env.config, outfile, indent=4, sort_keys=False)

                bit_stats_tt = env.qctrl.statistics()['Bitwidth distribution:']
                bit_stats_tt.set_max_width(100)
                bit_stats_df = pd.read_csv(
                    StringIO(re.sub(
                        r'[-+|=]', '', bit_stats_tt.draw())), sep=r'\s{2,}', engine='python').reset_index(drop=True)

                policy_dict[episode]=env.master_df['action'].astype('int')
                pd.DataFrame(
                    policy_dict.values(), index=policy_dict.keys()).T.sort_index(axis=1, ascending=False).to_csv(
                        osp.join(config['log_dir'], "policy_per_episode.csv"), index_label="nodestr")

                if final_reward > best_reward:
                    best_reward = final_reward
                    best_policy = env.master_df['action']

                    # log best policy to tensorboard
                    best_policy_dict[episode]=env.master_df['action'].astype('int')
                    pd.DataFrame(
                        best_policy_dict.values(), index=best_policy_dict.keys()).T.sort_index(
                            axis=1, ascending=False).to_csv(
                                osp.join(config['log_dir'], "best_policy.csv"), index_label="nodestr")

                    best_policy_string = bit_stats_df.to_markdown() + "\n\n\n"
                    best_policy_string += "Episode: {}, Reward: {:.3f}, Accuracy: {:.3f}, Model_Ratio: {:.3f}\n\n\n" \
                                          .format(episode, final_reward, info['accuracy'], info['model_ratio'])
                    for i, nodestr in enumerate(env.master_df.index.tolist()):
                        best_policy_string += "\t"
                        Qtype=' (WQ)' if env.master_df.is_wt_quantizer[nodestr] else ' (AQ)'

                        if env.skip_constraint is True:
                            best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + \
                                                  " | " + nodestr + Qtype + "  \n"
                        else:
                            if env.master_df.loc[nodestr, 'action'] == \
                               env.master_df.loc[nodestr, 'unconstrained_action']:
                                best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + \
                                                      " | " + nodestr + Qtype + "  \n"
                            else:
                                best_policy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " <= " + \
                                                      str(int(env.master_df.loc[nodestr, 'unconstrained_action'])) + \
                                                      " | " + nodestr + Qtype + "  \n"
                    tfwriter.add_text('AutoQ/best_policy', best_policy_string, episode)

                # log current policy to tensorboard
                current_strategy_string = bit_stats_df.to_markdown() + "\n\n\n"
                current_strategy_string += "Episode: {}, Reward: {:.3f}, Accuracy: {:.3f}, Model_Ratio: {:.3f}\n\n\n" \
                                           .format(episode, final_reward, info['accuracy'], info['model_ratio'])
                for i, nodestr in enumerate(env.master_df.index.tolist()):
                    current_strategy_string += "\t"
                    Qtype=' (WQ)' if env.master_df.is_wt_quantizer[nodestr] else ' (AQ)'
                    if env.skip_constraint is True:
                        current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + \
                                                   " | " + nodestr + Qtype + "  \n"
                    else:
                        if env.master_df.loc[nodestr, 'action'] == env.master_df.loc[nodestr, 'unconstrained_action']:
                            current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + \
                                                       " | " + nodestr + Qtype + "  \n"
                        else:
                            current_strategy_string += str(int(env.master_df.loc[nodestr, 'action'])) + " <= " + \
                                                       str(int(env.master_df.loc[nodestr, 'unconstrained_action'])) + \
                                                       " | " + nodestr + Qtype + "  \n"
                tfwriter.add_text('AutoQ/current_policy', current_strategy_string, episode)

                value_loss = agent.get_value_loss()
                policy_loss = agent.get_policy_loss()
                delta = agent.get_delta()

                tfwriter.add_scalar('AutoQ/reward/last', final_reward, episode)
                tfwriter.add_scalar('AutoQ/reward/best', best_reward, episode)
                tfwriter.add_scalar('AutoQ/accuracy', info['accuracy'], episode)
                tfwriter.add_scalar('AutoQ/model_ratio', info['model_ratio'], episode)
                tfwriter.add_scalar('AutoQ/agent/value_loss', value_loss, episode)
                tfwriter.add_scalar('AutoQ/agent/policy_loss', policy_loss, episode)
                tfwriter.add_scalar('AutoQ/agent/delta', delta, episode)

                logger.info('best reward: {}\n'.format(best_reward))
                logger.info('best policy: {}\n'.format(best_policy))

                episode_elapsed = time.time() - episode_start_ts
                logger.info('\n### Episode[{}] Elapsed: {:.3f}\n'.format(episode-1, episode_elapsed))

        return best_policy, best_reward
