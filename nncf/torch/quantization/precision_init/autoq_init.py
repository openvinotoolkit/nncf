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

import json
import math
import os
import os.path as osp
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nncf.common.hardware.config import HWConfigType
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.utils.debug import is_debug
from nncf.common.utils.os import safe_open
from nncf.config.schemata.defaults import AUTOQ_EVAL_SUBSET_RATIO
from nncf.config.schemata.defaults import AUTOQ_ITER_NUMBER
from nncf.config.schemata.defaults import AUTOQ_WARMUP_ITER_NUMBER
from nncf.config.schemata.defaults import PRECISION_INIT_BITWIDTHS
from nncf.torch.automl.agent.ddpg.ddpg import DDPG
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.torch.structures import AutoQPrecisionInitArgs


class AutoQPrecisionInitParams(BasePrecisionInitParams):
    def __init__(
        self,
        user_init_args: AutoQPrecisionInitArgs,
        dump_autoq_data: bool = False,
        iter_number: int = 0,
        warmup_iter_number: int = None,
        compression_ratio: float = None,
        eval_subset_ratio: float = None,
        ddpg_hparams_dict: Dict = None,
        hw_cfg_type: HWConfigType = None,
        skip_constraint: bool = False,
        finetune: bool = False,
        bits: List[int] = None,
    ):
        super().__init__(user_init_args)
        self.dump_autoq_data = dump_autoq_data
        self.iter_number = iter_number
        self.compression_ratio = compression_ratio
        self.eval_subset_ratio = eval_subset_ratio
        self.warmup_iter_number = warmup_iter_number
        if ddpg_hparams_dict is None:
            self.ddpg_hparams_dict = {}
        else:
            self.ddpg_hparams_dict = ddpg_hparams_dict
        self.hw_cfg_type = hw_cfg_type
        self.skip_constraint = skip_constraint
        self.finetune = finetune
        self.bits = bits

    @classmethod
    def from_config(
        cls,
        autoq_init_config_dict: Dict,
        user_init_args: AutoQPrecisionInitArgs,
        target_hw_config_type: Optional[HWConfigType],
    ) -> "AutoQPrecisionInitParams":
        dict_copy = autoq_init_config_dict.copy()
        dump_autoq_data = dict_copy.pop("dump_init_precision_data", False)
        iter_number = dict_copy.pop("iter_number", AUTOQ_ITER_NUMBER)
        compression_ratio = dict_copy.pop("compression_ratio", 0.15)
        eval_subset_ratio = dict_copy.pop("eval_subset_ratio", AUTOQ_EVAL_SUBSET_RATIO)
        warmup_iter_number = dict_copy.pop("warmup_iter_number", AUTOQ_WARMUP_ITER_NUMBER)
        skip_constraint = dict_copy.pop("skip_constraint", False)
        finetune = dict_copy.pop("finetune", False)
        bits = dict_copy.pop("bits", PRECISION_INIT_BITWIDTHS)

        return cls(
            user_init_args=user_init_args,
            dump_autoq_data=dump_autoq_data,
            iter_number=iter_number,
            warmup_iter_number=warmup_iter_number,
            ddpg_hparams_dict=dict_copy,
            hw_cfg_type=target_hw_config_type,
            compression_ratio=compression_ratio,
            eval_subset_ratio=eval_subset_ratio,
            skip_constraint=skip_constraint,
            finetune=finetune,
            bits=bits,
        )


class AutoQPrecisionInitializer(BasePrecisionInitializer):
    def __init__(
        self,
        algo: "ExperimentalQuantizationController",  # noqa: F821
        params: AutoQPrecisionInitParams,
        hw_precision_constraints: HardwareQuantizationConstraints,
    ):
        super().__init__(algo, params, hw_precision_constraints)
        self.quantization_controller = algo
        self._params = params
        self._init_args = params.user_init_args
        self._dump_autoq_data = params.dump_autoq_data
        self._iter_number = params.iter_number
        self._warmup_iter_number = params.warmup_iter_number
        self._ddpg_hparams_override = params.ddpg_hparams_dict
        self._hw_cfg_type = params.hw_cfg_type

    def apply_init(self) -> SingleConfigQuantizerSetup:
        from nncf.common.utils.debug import DEBUG_LOG_DIR
        from nncf.torch.automl.environment.quantization_env import QuantizationEnv

        if self._dump_autoq_data or is_debug():
            dump_dir = self._init_args.config.get("log_dir", None)
            if dump_dir is None:
                dump_dir = DEBUG_LOG_DIR
            self.dump_dir = Path(dump_dir) / Path("autoq") / Path("autoq_agent_dump")
            self.dump_dir.mkdir(parents=True, exist_ok=True)

            self.policy_dict = OrderedDict()  # key: episode
            self.best_policy_dict = OrderedDict()  # key: episode

            self._init_args.config["episodic_nncfcfg"] = str(self.dump_dir / "episodic_nncfcfg")
            os.makedirs(self._init_args.config["episodic_nncfcfg"], exist_ok=True)

            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(self.dump_dir)
                # log compression config to tensorboard
                self.tb_writer.add_text(
                    "AutoQ/run_config",
                    json.dumps(self._init_args.config["compression"], indent=4, sort_keys=False).replace("\n", "\n\n"),
                    0,
                )
            except ModuleNotFoundError:
                nncf_logger.warning(
                    "Tensorboard installation not found! Install tensorboard Python package "
                    "in order for AutoQ tensorboard statistics data to be dumped"
                )

        start_ts = datetime.now()

        from nncf.torch.automl.environment.quantization_env import QuantizationEnvParams

        env_params = QuantizationEnvParams(
            compression_ratio=self._params.compression_ratio,
            eval_subset_ratio=self._params.eval_subset_ratio,
            skip_constraint=self._params.skip_constraint,
            performant_bw=True,
            finetune=self._params.finetune,
            bits=self._params.bits,
            dump_init_precision_data=self._dump_autoq_data,
            log_dir=Path(DEBUG_LOG_DIR) / Path("autoq"),
        )

        # Instantiate Quantization Environment
        env = QuantizationEnv(
            self._model,
            self.quantization_controller,
            self._hw_precision_constraints,
            self._init_args.data_loader,
            self._init_args.eval_fn,
            hw_config_type=self._hw_cfg_type,
            params=env_params,
        )

        nb_state = len(env.state_list)
        nb_action = 1

        # Control buffer length at run manager level
        if "warmup_iter_number" not in self._ddpg_hparams_override:
            self._ddpg_hparams_override["warmup_iter_number"] = self._warmup_iter_number
        self._ddpg_hparams_override["rmsize"] = self._warmup_iter_number * (len(env.master_df) + 1)

        # Instantiate Automation Agent
        agent = DDPG(nb_state, nb_action, self._iter_number, hparam_override=self._ddpg_hparams_override)

        if self._dump_autoq_data and self.tb_writer is not None:
            # Need to replace '|' in nodestr (QuantizerId/QuantizerPointId)
            # to '+' as it is a special character in markdown
            temp_df = deepcopy(env.master_df[env.state_list + ["n_op"]])
            temp_df["modified_nodestr"] = list(map(lambda x: x.replace("|", "+"), temp_df.index.tolist()))
            temp_df = temp_df.set_index("modified_nodestr").reset_index()
            self.tb_writer.add_text("AutoQ/state_embedding", temp_df.to_markdown())

        best_policy, best_reward = self._search(agent, env)

        end_ts = datetime.now()

        final_qid_vs_qconfig_map = env.select_config_for_actions(best_policy)

        final_quantizer_setup = self.quantization_controller.get_quantizer_setup_for_current_state()
        for qp_id, qconf in final_qid_vs_qconfig_map.items():
            final_quantizer_setup.quantization_points[qp_id].qconfig = qconf

        str_bw = [str(element) for element in self.get_bitwidth_per_scope(final_quantizer_setup)]
        nncf_logger.info("\n".join(['[AutoQ]\n"bitwidth_per_scope": [', ",\n".join(str_bw), "]"]))
        nncf_logger.info(f"[AutoQ] best_reward: {best_reward}")
        nncf_logger.info(f"[AutoQ] best_policy: {best_policy}")
        nncf_logger.info("[AutoQ] Search completed.")
        nncf_logger.info("[AutoQ] Elapsed time of AutoQ Precision Initialization (): {}".format(end_ts - start_ts))
        return final_quantizer_setup

    def _search(self, agent: DDPG, env: "QuantizationEnv") -> Tuple[pd.Series, float]:  # noqa: F821
        best_reward = -math.inf
        episode = 0
        episode_reward = 0.0
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
                nncf_logger.info(
                    f"## Episode[{episode}], "
                    f"reward: {episode_reward:.3f}, "
                    f'acc: {info["accuracy"]:.3f}, '
                    f'model_ratio: {info["model_ratio"]:.3f}, '
                    f'model_size(MB): {info["model_size"] / 8e6:.2f}, '
                    f'BOP_ratio: {info["bop_ratio"]:.3f}\n'
                )

                # Replay Buffer Management
                if agent.memory.nb_entries % (len(env.master_df) + 1) > 0:
                    raise ValueError("logical bug in buffer management, uneven episode length")
                if agent.memory.limit % (len(env.master_df) + 1) > 0:
                    raise ValueError("replay buffer size must be divisible by episode step length")

                if agent.memory.nb_entries + len(transition_buffer) >= agent.memory.limit:
                    step_reward_per_episode = agent.memory.rewards.data[:: (len(transition_buffer) + 1)]
                    sorted_index_of_episodes = np.argsort(step_reward_per_episode)  # ascending order

                    # Retain the top 30% of highest rewarded episodes,
                    # discard by sampling an episode uniformly from the lower 70% episodes
                    discard_candidates = sorted_index_of_episodes[: int(len(sorted_index_of_episodes) * 0.7)]
                    discard_episode = np.random.choice(discard_candidates)

                    discard_start_index = (discard_episode) * (len(transition_buffer) + 1)
                    discard_end_index = (discard_episode + 1) * (len(transition_buffer) + 1)

                    agent.memory.discard(slice(discard_start_index, discard_end_index))
                # = EO Replay Buffer Management

                final_reward = transition_buffer[-1][0]
                r_per_step = final_reward / len(env.master_df)

                for i, (_, s_t, _, a_t, done) in enumerate(transition_buffer):
                    # Revision of prev_action as it could be modified by constrainer -------
                    if i == 0:
                        prev_action = 0.0
                    else:
                        prev_action = env.master_df["action"][i - 1] / 8  # ducktape scaling
                    if prev_action != s_t["prev_action"]:
                        s_t["prev_action"] = prev_action
                    # EO ------------------------
                    agent.observe(r_per_step, s_t, a_t, done)

                agent.memory.append(observation, agent.select_action(observation, episode=episode), 0.0, False)

                # update DDPG networks, note that this loop must not be
                # in the loop above to avoid non-numeric value in replay buffer
                for i, (_, s_t, _, a_t, done) in enumerate(transition_buffer):
                    if episode >= agent.warmup_iter_number:
                        for _ in range(agent.n_update):
                            agent.update_policy()

                # reset
                observation = None
                episode_reward = 0.0
                transition_buffer = []

                value_loss = agent.get_value_loss()
                policy_loss = agent.get_policy_loss()
                delta = agent.get_delta()

                nncf_stats = env.qctrl.statistics()
                bit_stats_df = (
                    pd.DataFrame.from_dict(
                        [nncf_stats.quantization.num_wq_per_bitwidth, nncf_stats.quantization.num_aq_per_bitwidth]
                    )
                    .fillna(0)
                    .astype(int)
                    .rename(index={0: "WQ", 1: "AQ"})
                    .transpose()
                    .sort_index(ascending=False)
                )
                bit_stats_df.index.name = "bitwidth"
                bit_stats_df = bit_stats_df.reset_index()

                if final_reward > best_reward:
                    best_reward = final_reward
                    best_policy = deepcopy(env.master_df["action"])
                    info_tuple = (episode, best_reward, info["accuracy"], info["model_ratio"], info["bop_ratio"])
                    self._dump_best_episode(info_tuple, bit_stats_df, env)
                    log_str = (
                        f"## Episode[{episode}] "
                        f"New best policy: {best_policy.values.tolist()}, "
                        f"reward: {best_reward:.3f}, "
                        f'acc: {info["accuracy"]:.3f}, '
                        f'model_ratio: {info["model_ratio"]:.3f},'
                        f' BOP_ratio: {info["bop_ratio"]:.3f}'
                    )
                    nncf_logger.info(f"\033[92m {log_str}\033[00m")

                episodic_info_tuple = (
                    episode,
                    final_reward,
                    best_reward,
                    info["accuracy"],
                    info["model_ratio"],
                    info["bop_ratio"],
                    value_loss,
                    policy_loss,
                    delta,
                )
                self._dump_episode(episodic_info_tuple, bit_stats_df, env, agent)

                episode_elapsed = time.time() - episode_start_ts

                nncf_logger.info(f'## Episode[{episode}] Policy: \n{env.master_df["action"].to_string()}\n')
                nncf_logger.info(f"## Episode[{episode}] Elapsed: {episode_elapsed:.3f}\n")

                episode += 1

        return best_policy, best_reward

    def _dump_best_episode(self, info_tuple: Tuple, bit_stats_df: pd.DataFrame, env: "QuantizationEnv"):  # noqa: F821
        if self._dump_autoq_data:
            episode = info_tuple[0]
            self.best_policy_dict[episode] = env.master_df["action"].astype("int")

            pd.DataFrame(self.best_policy_dict.values(), index=self.best_policy_dict.keys()).T.sort_index(
                axis=1, ascending=False
            ).to_csv(osp.join(self.dump_dir, "best_policy.csv"), index_label="nodestr")

            best_policy_string = self._generate_tensorboard_logging_string(
                bit_stats_df, env.master_df, info_tuple, env.skip_constraint
            )

            list_of_dump_dict = []
            for i, _ in enumerate(env.groups_of_adjacent_quantizers):
                list_of_dump_dict.append(env.master_df.loc[env.adjq_groupwise_df_lut_keys[i], ["action"]].to_dict())
            best_policy_string += (
                "\t\n\t# Precision(s) per Group of Adjacent Quantizers\n\t"
                + json.dumps(list_of_dump_dict, indent=4).replace("\n", "\n\t")
                + "\n\n"
            )

            self.tb_writer.add_text("AutoQ/best_policy", best_policy_string, episode)

    def _dump_episode(
        self, episodic_info_tuple: Tuple, bit_stats_df: pd.DataFrame, env: "QuantizationEnv", agent: DDPG  # noqa: F821
    ):
        if self._dump_autoq_data:
            episode, final_reward, _, accuracy, model_ratio, bop_ratio, _, _, _ = episodic_info_tuple

            current_bitwidth_per_scope = self.get_bitwidth_per_scope(env.qctrl.get_quantizer_setup_for_current_state())

            current_episode_nncfcfg = deepcopy(self._init_args.config)
            current_episode_nncfcfg["compression"]["initializer"]["precision"] = {
                "bitwidth_per_scope": current_bitwidth_per_scope
            }

            # Save nncf compression cfg
            episode_cfgfile = "{0}/{1:03d}_nncfcfg.json".format(
                str(self._init_args.config["episodic_nncfcfg"]), episode
            )

            with safe_open(Path(episode_cfgfile), "w") as outfile:
                json.dump(current_episode_nncfcfg, outfile, indent=4, sort_keys=False)

            self.policy_dict[episode] = env.master_df["action"].astype("int")
            pd.DataFrame(self.policy_dict.values(), index=self.policy_dict.keys()).T.sort_index(
                axis=1, ascending=False
            ).to_csv(osp.join(self.dump_dir, "policy_per_episode.csv"), index_label="nodestr")

            # log current episode policy and feedback as text
            info_tuple = (episode, final_reward, accuracy, model_ratio, bop_ratio)
            current_strategy_string = self._generate_tensorboard_logging_string(
                bit_stats_df, env.master_df, info_tuple, env.skip_constraint
            )

            if env.performant_bw is True:
                list_of_dump_dict = []
                for i, _ in enumerate(env.groups_of_adjacent_quantizers):
                    list_of_dump_dict.append(
                        env.master_df.loc[env.adjq_groupwise_df_lut_keys[i], ["action", "action_aligned"]].to_dict()
                    )
                current_strategy_string += (
                    "\t\n\t# Precision(s) per Group of Adjacent Quantizers\n\t"
                    + json.dumps(list_of_dump_dict, indent=4).replace("\n", "\n\t")
                    + "\n\n"
                )

            self.tb_writer.add_text("AutoQ/current_policy", current_strategy_string, episode)

            # visualization over episode
            if self.tb_writer is not None:
                self._add_to_tensorboard(self.tb_writer, episodic_info_tuple)

            if episode % int((self._iter_number + 10) / 10) == 0:
                agent.save_model(self.dump_dir)

    def _add_to_tensorboard(self, tb_writer: "SummaryWriter", log_tuple: Tuple):  # noqa: F821
        episode, final_reward, best_reward, accuracy, model_ratio, bop_ratio, value_loss, policy_loss, delta = log_tuple

        tb_writer.add_scalar("AutoQ/reward/last", final_reward, episode)
        tb_writer.add_scalar("AutoQ/reward/best", best_reward, episode)
        tb_writer.add_scalar("AutoQ/accuracy", accuracy, episode)
        tb_writer.add_scalar("AutoQ/model_ratio", model_ratio, episode)
        tb_writer.add_scalar("AutoQ/bop_ratio", bop_ratio, episode)
        tb_writer.add_scalar("AutoQ/agent/value_loss", value_loss, episode)
        tb_writer.add_scalar("AutoQ/agent/policy_loss", policy_loss, episode)
        tb_writer.add_scalar("AutoQ/agent/delta", delta, episode)

    def _generate_tensorboard_logging_string(
        self, bit_stats_df: pd.DataFrame, master_df: pd.DataFrame, info_tuple: Tuple, skip_constraint=False
    ) -> str:
        qdf = master_df  # For readibility
        episode, reward, accuracy, model_ratio, bop_ratio = info_tuple

        text_string = bit_stats_df.to_markdown() + "\n\n\n"
        text_string += "Episode: {:>4}, Reward: {:.3f}, ".format(episode, reward)
        text_string += "Accuracy: {:.3f}, Model_Size_Ratio: {:.3f}, BOP_Ratio: {:.3f}\n\n\n".format(
            accuracy, model_ratio, bop_ratio
        )

        for _, row_id in enumerate(qdf.index.tolist()):
            Qtype = "(WQ)" if qdf.is_wt_quantizer[row_id] else "(AQ)"

            if skip_constraint is False and qdf.loc[row_id, "action"] != qdf.loc[row_id, "unconstrained_action"]:
                text_string += "\t{} <= {} | {} {} \n".format(
                    str(int(qdf.loc[row_id, "action"])),
                    str(int(qdf.loc[row_id, "unconstrained_action"])),
                    Qtype,
                    row_id,
                )

            else:
                text_string += "\t{} | {} {} \n".format(str(int(qdf.loc[row_id, "action"])), Qtype, row_id)

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
