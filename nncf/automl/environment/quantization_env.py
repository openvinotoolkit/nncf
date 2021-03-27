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

import logging

import os.path as osp
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import ctypes
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from natsort import natsorted

from nncf.debug import is_debug, DEBUG_LOG_DIR
from nncf.common.utils.logger import logger
from nncf.hw_config import HWConfigType
from nncf.initialization import PartialDataLoader
from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.algo import QuantizationController, NNCFNetwork, ExperimentalQuantizationController
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.quantization.quantizer_id import QuantizerId, WeightQuantizerId, \
    NonWeightQuantizerId, InputQuantizerId, FunctionQuantizerId
from nncf.common.quantization.structs import QuantizerConfig

from sklearn.preprocessing import MinMaxScaler

from nncf.quantization.quantizer_setup import QuantizationPointId
from nncf.common.os import safe_open
from nncf.quantization.precision_init.compression_ratio import CompressionRatioCalculator

def find_qid_by_str(qctrl: QuantizationController, qid_str: str) -> QuantizerId:
    for _qid, _q in qctrl.all_quantizations.items():
        if qid_str == str(_qid):
            return _qid
    return None

class ModelSizeCalculator:
    FLOAT_BITWIDTH = ctypes.sizeof(ctypes.c_float) * 8
    def __init__(self, qmodel: NNCFNetwork, per_quantizer_config_space: Dict[QuantizerId, List[QuantizerConfig]]):
        self._bw_space_map = OrderedDict()
        self._nparam_map = OrderedDict()
        for qid, qconfig_space in per_quantizer_config_space.items():
            if isinstance(qid, WeightQuantizerId):
                self._bw_space_map[qid] = [qconf.num_bits for qconf in qconfig_space]
                m = qmodel.get_module_by_scope(qid.scope)
                self._nparam_map[qid] = np.prod(m.weight.size())

        self.min_model_size, self.max_model_size = \
            self._calc_min_max_model_size()

        self.fp_model_size = self.get_uniform_bit_model_size(ModelSizeCalculator.FLOAT_BITWIDTH)

    def __call__(self, per_quantizer_bw):
        return self.get_model_size(per_quantizer_bw)

    def _calc_min_max_model_size(self) -> Tuple[np.int64, np.int64]:
        nparam_array = np.array(list(self._nparam_map.values()))
        min_size = np.sum(nparam_array * np.array(list(map(min, self._bw_space_map.values()))))
        max_size = np.sum(nparam_array * np.array(list(map(max, self._bw_space_map.values()))))
        return min_size, max_size

    def get_uniform_bit_model_size(self, uniform_bitwidth: int) -> np.int64:
        return np.sum(np.array(list(self._nparam_map.values())*uniform_bitwidth))

    def get_model_size(self, per_quantizer_bw: Dict[QuantizerId, int]) -> np.int64:
        model_size = 0
        for qid, nparam in self._nparam_map.items():
            if qid in per_quantizer_bw:
                model_size += nparam * per_quantizer_bw[qid]
            else:
                logger.warning("[ModelSizeCalculator] Missing Bitwidth of QID: {}, using {} bits"
                               .format(str(qid), ModelSizeCalculator.FLOAT_BITWIDTH))
                model_size += nparam * ModelSizeCalculator.FLOAT_BITWIDTH
        return model_size

    def get_model_size_ratio(self, per_quantizer_bw: Dict[QuantizerId, int]) -> np.float64:
        return self.get_model_size(per_quantizer_bw)/self.fp_model_size

class QuantizationEnvParams:
    def __init__(self, compression_ratio: float,
                 eval_subset_ratio: float,
                 skip_constraint: bool,
                 finetune: bool,
                 bits: List[int],
                 dump_init_precision_data: bool = False,
                 log_dir: str = None):
        self.compression_ratio = compression_ratio
        self.eval_subset_ratio = eval_subset_ratio
        self.skip_constraint = skip_constraint
        self.finetune = finetune
        self.bits = bits
        self.dump_init_precision_data = dump_init_precision_data
        self.log_dir = log_dir


class QuantizationEnv:
    # pylint:disable=too-many-branches,too-many-statements
    def __init__(self,
                 model: NNCFNetwork,
                 quantization_controller: ExperimentalQuantizationController,
                 hw_precision_constraints: HardwareQuantizationConstraints,
                 eval_loader: torch.utils.data.DataLoader,
                 eval_fn: Callable[[nn.Module, torch.utils.data.DataLoader], float],
                 hw_config_type: HWConfigType,
                 params: QuantizationEnvParams
                 ):

        logger.info("[Q.Env] Instantiating NNCF Quantization Environment")
        self.qctrl = quantization_controller
        self.qmodel = model
        self.eval_loader = eval_loader
        self.eval_fn = eval_fn
        self._hw_precision_constraints = hw_precision_constraints

        self.model_name = self.qmodel.nncf_module.__class__.__name__

        # Check and only proceed if target device is supported by Q.Env
        self.hw_cfg_type = hw_config_type
        assert self.hw_cfg_type in [None, HWConfigType.VPU]

        # Set target compression ratio
        self.compression_ratio = params.compression_ratio

        self.eval_loader = PartialDataLoader(self.eval_loader, iter_ratio=params.eval_subset_ratio)

        # Bool to disable hard resource constraint
        self.skip_constraint = params.skip_constraint

        # Bool to enable fine-tuning in each episode. Placeholder for now
        self.finetune = False

        # Configure search space for precision according to target device
        if self.hw_cfg_type is None:
            self.model_bitwidth_space = params.bits
        elif self.hw_cfg_type is HWConfigType.VPU:
            self.model_bitwidth_space = self._hw_precision_constraints.get_all_unique_bitwidths()
        self.model_bitwidth_space = sorted(list(self.model_bitwidth_space))

        # Create mapping of QuantizerId to the space of the corresponding quantizer's allowed qconfigs
        #pylint:disable=line-too-long
        self.qconfig_space_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())  # type: Dict[QuantizerId, List[QuantizerConfig]]
        if self.hw_cfg_type is None:
            for qid in self.qconfig_space_map.keys():
                conf = self.qctrl.all_quantizations[qid].get_quantizer_config()
                conf_list_to_set = []
                for bit in self.model_bitwidth_space:
                    bit_adjusted_conf = deepcopy(conf)
                    bit_adjusted_conf.num_bits = bit
                    conf_list_to_set.append(bit_adjusted_conf)
                self.qconfig_space_map[qid] = conf_list_to_set
        else:
            for qid in self.qconfig_space_map:
                conf_list_to_set = []
                bw_vs_qconfigs_dict = self._hw_precision_constraints.get_bitwidth_vs_qconfigs_dict(qid)
                for bitwidth, qconf_list in bw_vs_qconfigs_dict.items():
                    target_qconf = qconf_list[0]
                    if len(qconf_list) > 1:
                        logger.warning("Received multiple quantizer configurations {qc_lst} for same bitwidth {bw} "
                                       "for quantizer {q} - AutoQ can currently only choose among bitwidths, but not "
                                       "within quantizer configuration space with the same bitwidths. Selecting {qc} "
                                       "as the target configuration for bitwidth {bw}".format(
                            qc_lst=";".join([str(qconf) for qconf in qconf_list]),
                            bw=bitwidth,
                            q=str(qid),
                            qc=str(target_qconf)))
                    conf_list_to_set.append(target_qconf)

                self.qconfig_space_map[qid] = conf_list_to_set

        # Quantizer Master Table Creation
        self._groups_of_adjacent_quantizers = self.qctrl._groups_of_adjacent_quantizers
        self.quantizer_table = self._create_quantizer_table()

        # Create master dataframe to keep track of quantizable layers and their attributes
        self.master_df, self.state_list = self._get_state_space(self.qctrl, self.qmodel, self.quantizer_table)
        if self.master_df.isnull().values.any():
            raise ValueError("Q.Env Master Dataframe has null value(s)")

        assert len(self.quantizer_table) == len(self.qctrl.all_quantizations), \
            "Number of Quantizer is not tally between quantizer table and quantization controller"

        # MinMaxScaler for State Embedding
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.master_df[self.state_list])

        # Model Size Calculation
        self.model_size_calculator = ModelSizeCalculator(self.qmodel, self.qconfig_space_map)
        self.orig_model_size = self.model_size_calculator.fp_model_size
        self.min_model_size = self.model_size_calculator.min_model_size
        self.max_model_size = self.model_size_calculator.max_model_size
        self.target_model_size = self.orig_model_size * self.compression_ratio

        if self.target_model_size < self.min_model_size and self.target_model_size > self.max_model_size:
            raise ValueError("Model Size Ratio {} is out of bound ({}, {})"
                             .format(self.compression_ratio,
                                     self.min_model_size / self.orig_model_size,
                                     self.max_model_size / self.orig_model_size))

        # Compression Ratio Calculation (BOP relative to 8-bit)
        self.compression_ratio_calculator = CompressionRatioCalculator(
            self.qmodel.get_flops_per_module(),
            self.qctrl.get_quantizer_setup_for_current_state(),
            self.qctrl.groups_of_adjacent_quantizers.weight_qp_id_per_activation_qp_id)

        # Evaluate and store metric score of pretrained model
        self._evaluate_pretrained_model()
        self.qmodel_init_sd = deepcopy(self.qmodel.state_dict())

        self.reset()

        self._dump_autoq_data = params.dump_init_precision_data
        if self._dump_autoq_data or is_debug():
            dump_dir = params.log_dir
            if dump_dir is None:
                dump_dir = DEBUG_LOG_DIR
            self.dump_dir = Path(dump_dir) / Path("autoq_env_dump")
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            # Serialize Q.Env information. Note that these functions should be at the end of Q.Env Initialization.
            self._dump_master_df()
            self._dump_quantized_graph()
            self._dump_groups_of_adjacent_quantizers()

        # End of QuantizationEnv.__init__()
        # --------------------------------------------------------------------------------------------------------------


    def reset(self):
        self.collected_strategy = []
        self.master_df['action'] = max(self.model_bitwidth_space)
        self.master_df['prev_action'] = 0
        self.master_df['unconstrained_action'] = 0


    def _create_quantizer_table(self) -> pd.DataFrame:
        # Create a mapping of qid to its adjacent quantizer group id
        adjq_gid_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())
        for qid in self.qctrl.all_quantizations:
            adjq_gid_map[qid] = self._groups_of_adjacent_quantizers.get_group_id_for_quantizer(qid)

        assert len(set(self.qconfig_space_map.keys()) - set(adjq_gid_map.keys())) == 0, \
            "both qconfig_space_map and adjq_gid_map must have exact keys."

        # Create a mapping of qid to its nodekey in NNCFGraph
        qid_nodekey_map = self._generate_qid_nodekey_map(self.qctrl, self.qmodel)

        assert len(set(qid_nodekey_map.keys()) - set(adjq_gid_map.keys())) == 0, \
            "both qid_nodekey_map and adjq_gid_map must have exact keys."

        d = OrderedDict()
        for qid in self.qctrl.all_quantizations:
            nodekey = qid_nodekey_map[qid]
            q_nx_nodeid = nodekey.split()[0]
            idx_str = '-'.join([q_nx_nodeid, str(qid)])
            gid = adjq_gid_map[qid]

            d[idx_str] = OrderedDict()
            d[idx_str]['qid'] = str(qid)
            d[idx_str]['q_nx_nodeid'] = q_nx_nodeid
            d[idx_str]['q_nx_nodekey'] = nodekey
            d[idx_str]['gid'] = gid
            d[idx_str]['qconf_space'] = self.qconfig_space_map[qid]
            d[idx_str]['qp_id_set'] = self.qctrl.module_id_to_qp_id_translation_dict[qid]

            if isinstance(qid, WeightQuantizerId):
                d[idx_str]['state_scope'] = qid.scope
            elif isinstance(qid, NonWeightQuantizerId):
                d[idx_str]['state_scope'] = qid.ia_op_exec_context.scope_in_model
            else:
                raise NotImplementedError("QuantizerId: {} of {} class is not supported."
                                          .format(str(qid), str(qid.__class__.__name__)))

        # quantizer_table index is QuantizerId in string prepended with its quantize node id in NNCFGraph
        df = pd.DataFrame.from_dict(d, orient='index')
        df['qid_obj'] = df['qid'].apply(lambda x: find_qid_by_str(self.qctrl, x))
        df['qmodule'] = df['qid_obj'].apply(lambda x: self.qctrl.all_quantizations[x])
        df['is_wt_quantizer'] = df['qid_obj'].apply(lambda x: x in self.qctrl.weight_quantizers)
        df['state_module'] = df['state_scope'].apply(self.qmodel.get_module_by_scope)

        quantizer_table = df.loc[natsorted(df.index)]
        return quantizer_table


    def _get_state_space(self,
                         quantization_controller: QuantizationController,
                         quantized_model: NNCFNetwork,
                         quantizer_table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        def annotate_learnable_module_io_shape(model):
            def annotate_io_shape(module, input_, output):
                if hasattr(module, 'weight') or isinstance(module, BaseQuantizer):
                    module.input_shape_ = input_[0].shape
                    module.output_shape_ = output.shape

            hook_list = [m.register_forward_hook(annotate_io_shape) for n, m in model.named_modules()]
            model.do_dummy_forward(force_eval=True)
            for h in hook_list:
                h.remove()

        annotate_learnable_module_io_shape(quantized_model)

        # State Embedding Extraction
        #---------------------------
        df = quantizer_table
        layer_attr_df = df.apply(self._get_layer_attr, axis=1)
        layer_attr_df['layer_idx'] = np.array(range(len(layer_attr_df)))
        layer_attr_df['weight_quantizer'] = df['is_wt_quantizer'].astype('float')
        state_list = layer_attr_df.columns.to_list()

        # create master dataframe
        master_df = pd.concat([df, layer_attr_df], axis='columns')

        # Annotate a min and a max value in prev_action before minmaxscaler fitting
        master_df.loc[master_df.index[0], 'prev_action'] = max(self.model_bitwidth_space)
        master_df.loc[master_df.index[-1], 'prev_action'] = min(self.model_bitwidth_space)

        # add GEMM Ops to weight quantizer
        master_df['n_op'] = master_df['state_scope'].map(self.qmodel.get_flops_per_module())
        master_df['n_op'] = master_df['n_op'].fillna(0)

        return master_df, state_list


    def _get_layer_attr(self, row: pd.Series) -> pd.Series:
        m = row.state_module
        qid = row.qid_obj
        feature = OrderedDict()

        if isinstance(qid, WeightQuantizerId):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                feature['conv_dw'] = int(m.weight.shape[1] == m.groups) # 1.0 for depthwise, 0.0 for other conv2d
                feature['cin'] = m.weight.shape[1]
                feature['cout'] = m.weight.shape[0]
                feature['stride'] = m.stride[0]
                feature['kernel'] = m.kernel_size[0]
                feature['param'] = np.prod(m.weight.size())
                feature['ifm_size'] = np.prod(m.input_shape_[-2:]) # H*W
                feature['prev_action'] = 0.0 # placeholder

            elif isinstance(m, nn.Linear):
                feature['conv_dw'] = 0.0
                feature['cin'] = m.in_features
                feature['cout'] = m.out_features
                feature['stride'] = 0.0
                feature['kernel'] = 1.0
                feature['param'] = np.prod(m.weight.size())
                feature['ifm_size'] = np.prod(m.input_shape_[-1]) # feature nodes
                feature['prev_action'] = 0.0 # placeholder

            else:
                raise NotImplementedError("State embedding extraction of {}".format(m.__class__.__name__))

        elif isinstance(qid, NonWeightQuantizerId):
            qmod = self.qctrl.all_quantizations[qid]
            input_shape = qmod.input_shape_
            output_shape = qmod.output_shape_
            feature['cin'] = input_shape[1] if len(input_shape) == 4 else input_shape[-1]
            feature['cout'] = output_shape[1] if len(output_shape) == 4 else output_shape[-1]
            feature['ifm_size'] = np.prod(input_shape[-2:]) if len(input_shape) == 4 else input_shape[-1]
            feature['conv_dw'] = 0.0
            feature['stride'] = 0.0
            feature['kernel'] = 0.0
            feature['param'] = 0.0
            feature['prev_action'] = 0.0

            if len(input_shape) != 4 and len(input_shape) != 2:
                raise NotImplementedError("A design is required to cater this scenario. Pls. report to maintainer")

        elif isinstance(qid, InputQuantizerId):
            raise NotImplementedError("InputQuantizerId is supported, quantizer of nncf model input "
                                      "is of type NonWeightQuantizerId")

        elif isinstance(qid, FunctionQuantizerId):
            raise NotImplementedError("FunctionQuantizerId is supported, Pls. report to maintainer")

        else:
            raise ValueError("qid is an instance of unexpected class {}".format(qid.__class__.__name__))

        return pd.Series(feature)


    def _evaluate_pretrained_model(self):
        logger.info("[Q.Env] Evaluating Pretrained Model")
        self.qctrl.disable_weight_quantization()
        self.qctrl.disable_activation_quantization()

        with torch.no_grad():
            self.pretrained_score = self.eval_fn(self.qmodel, self.eval_loader)
            logger.info("Pretrained Score: {:.3f}".format(self.pretrained_score))

        self.qctrl.enable_weight_quantization()
        self.qctrl.enable_activation_quantization()
        self.qmodel.rebuild_graph()


    def _run_quantization_pipeline(self, finetune=False) -> float:
        self.qctrl.run_batchnorm_adaptation(self.qctrl.quantization_config)

        if finetune:
            raise NotImplementedError("Post-Quantization fine tuning is not implemented.")
        with torch.no_grad():
            quantized_score = self.eval_fn(self.qmodel, self.eval_loader)
            logger.info("[Q.Env] Quantized Score: {:.3f}".format(quantized_score))
        return quantized_score


    def _get_quantizer_bitwidth(self) -> Dict[BaseQuantizer, int]:
        assert len(set(self.model_bitwidth_space) - set(self.master_df.action.values)) >= 0, \
            "there is bitwidth choice not within model bitwidth space"
        return OrderedDict(zip(self.master_df.qid_obj, self.master_df.action))


    def _constrain_model_size(self, collected_strategy: List, skip=False) -> List:
        def lower_bitwidth(bw: int, qconf_space: List[QuantizerConfig]) -> int:
            bw_space = [qconf.num_bits for qconf in qconf_space]
            assert bw in bw_space
            sorted_bw_space = sorted(bw_space)
            return sorted_bw_space[sorted_bw_space.index(bw)-1] if sorted_bw_space.index(bw) > 0 else bw

        # This function acts on self.master_df['action']
        self.master_df['action'] = collected_strategy

        if skip is not True:
            self.master_df['unconstrained_action'] = self.master_df['action']

            current_model_size = self.model_size_calculator(self._get_quantizer_bitwidth())

            while self.min_model_size < current_model_size and self.target_model_size < current_model_size:
                for _, nodestr in enumerate(reversed(self.master_df.index.tolist())):
                    if self.master_df.loc[nodestr, "is_wt_quantizer"]:
                        bw_choice, qconf_space = self.master_df.loc[nodestr, ['action', 'qconf_space']]
                        new_bw = lower_bitwidth(bw_choice, qconf_space)
                        self.master_df.loc[nodestr, "action"] = new_bw if new_bw != bw_choice else bw_choice

                    current_model_size = self.model_size_calculator(self._get_quantizer_bitwidth())
                    if current_model_size <= self.target_model_size:
                        break
        else:
            logger.info("[Q.Env] Skipping Model Size Constraint")

        return self.master_df['action'].tolist()

    def reward(self, acc: float, model_ratio: float) -> float:
        return (acc - self.pretrained_score) * 0.1

    def step(self, action: Union[int, float]) -> Tuple:
        currently_processed_qconf_idx = len(self.collected_strategy)

        def is_final_step():
            return len(self.collected_strategy) == len(self.master_df)

        # Ensure action is in the quantizer's bitwidth space
        current_qconf_space = self.master_df.qconf_space[currently_processed_qconf_idx]
        current_bw_space = [qconf.num_bits for qconf in current_qconf_space]
        if action not in current_bw_space:
            closest_bw_idx = np.argmin(np.abs(action - np.array(current_bw_space)))
            action = current_bw_space[closest_bw_idx]

        self.collected_strategy.append(action)

        if not is_final_step():
            info_set = {}
            reward = 0
            self.set_next_step_prev_action(len(self.collected_strategy), action)
            obs = self.get_normalized_obs(len(self.collected_strategy))
            done = False
            return obs, reward, done, info_set

        return self.evaluate_strategy(self.collected_strategy, skip_constraint=self.skip_constraint)


    def select_config_for_actions(self, actions) -> Dict[QuantizationPointId, QuantizerConfig]:
        retval = OrderedDict()  # type: Dict[QuantizationPointId, QuantizerConfig]
        for action, qp_id_set, qconf_space in zip(actions,
                                            self.master_df['qp_id_set'],
                                            self.master_df['qconf_space']):
            matches = []
            for qconf in qconf_space:
                if qconf.num_bits == action:
                    matches.append(qconf)
            assert len(matches) == 1
            for qp_id in qp_id_set:
                retval[qp_id] = matches[0]
        return retval

    def evaluate_strategy(self, collected_strategy: List, skip_constraint=True) -> Tuple:
        assert len(collected_strategy) == len(self.master_df)
        if skip_constraint is not True:
            collected_strategy = self._constrain_model_size(collected_strategy)
        self.master_df['action'] = collected_strategy # This must be after constraint

        configs_to_set = self.select_config_for_actions(self.master_df['action'])
        self._apply_quantizer_configs_to_model(configs_to_set)

        for idx, qid in zip(self.master_df.index, self.master_df['qid']):
            logger.info("[Q.Env] {:50} | {}".format(
                str(self.qctrl.all_quantizations[find_qid_by_str(self.qctrl, qid)]),
                idx))

        quantized_score = self._run_quantization_pipeline(finetune=self.finetune)

        current_model_size = self.model_size_calculator(self._get_quantizer_bitwidth())
        current_model_ratio = self.model_size_calculator.get_model_size_ratio(self._get_quantizer_bitwidth())

        current_model_bop_ratio = self.compression_ratio_calculator.run_for_quantizer_setup(self.qctrl.get_quantizer_setup_for_current_state())

        reward = self.reward(quantized_score, current_model_ratio)

        info_set = {'model_ratio': current_model_ratio, 'accuracy': quantized_score, 'model_size': current_model_size, 'bop_ratio': current_model_bop_ratio}

        obs = self.get_normalized_obs(len(collected_strategy) - 1)
        done = True

        return obs, reward, done, info_set


    def set_next_step_prev_action(self, idx, action):
        self.master_df.loc[self.master_df.index[idx], 'prev_action'] = action


    def get_normalized_obs(self, idx: int) -> pd.Series:
        _df = self.master_df.loc[self.master_df.index, self.state_list]
        _df.loc[_df.index, self.state_list] = self.state_scaler.transform(_df[self.state_list])
        return _df.iloc[idx]

    def _apply_quantizer_configs_to_model(self, qid_vs_qconfig_map: Dict[QuantizationPointId, QuantizerConfig]):
        new_quantizer_setup = self.qctrl.get_quantizer_setup_for_current_state()
        for qp_id, qconf in qid_vs_qconfig_map.items():
            new_quantizer_setup.quantization_points[qp_id].qconfig = qconf
        self.qmodel.load_state_dict(self.qmodel_init_sd)
        had_to_regenerate = self.qctrl.is_new_setup_requires_regeneration(new_quantizer_setup)
        self.qctrl, self.qmodel = self.qctrl.apply_new_quantizer_setup(new_quantizer_setup)
        if had_to_regenerate:
            self.qmodel_init_sd = deepcopy(self.qmodel.state_dict())

        # The QuantizerId's may have changed after the new quantizer setup application, but
        # QuantizationPointId's should not have. Will use this to update the qids in the master table.
        for qid, qp_id_set in self.qctrl.module_id_to_qp_id_translation_dict.items():
            self.master_df.loc[self.master_df.qp_id_set == qp_id_set].qid = str(qid)

    def _generate_qid_nodekey_map(self,
                                  quantization_controller: QuantizationController,
                                  quantized_network: NNCFNetwork) -> Dict[QuantizerId, str]:
        """
        Create a lookup mapping for each QuantizerId to its corresponding quantize node in network graph
        :param quantization_controller:
        :param quantized_network:
        :return: dict with key of QuantizerId and value of node key string
        """
        # Map Non Weight Qid to its nodes in nxgraph
        weight_quantize_nodekeys = []
        non_weight_quantize_nodekeys = []
        qid_nodekey_map = OrderedDict()

        quantized_network.rebuild_graph()
        g = quantized_network.get_graph()

        for nodekey in g.get_all_node_keys():
            if 'symmetric_quantize' in nodekey and 'UpdateWeight' in nodekey:
                weight_quantize_nodekeys.append(nodekey)
            if 'symmetric_quantize' in nodekey and 'UpdateWeight' not in nodekey:
                non_weight_quantize_nodekeys.append(nodekey)

        # Find nodekey of Weight Quantizer
        for qid, _ in quantization_controller.weight_quantizers.items():
            quantize_nodekeys = []
            for nodekey in weight_quantize_nodekeys:
                if str(qid.scope) in nodekey:
                    quantize_nodekeys.append(nodekey)

            if len(quantize_nodekeys) == 1:
                qid_nodekey_map[qid] = quantize_nodekeys[0]
            else:
                raise ValueError("Quantize Node not found or More Nodes are found for WQid: {}".format(qid))

        # Find nodekey of Non-Weight Quantizer
        for qid, _ in quantization_controller.non_weight_quantizers.items():
            quantize_nodekeys = []
            for nodekey in non_weight_quantize_nodekeys:
                if str(qid.ia_op_exec_context.scope_in_model) in nodekey:
                    quantize_nodekeys.append(nodekey)

            if len(quantize_nodekeys) > 0:
                qid_nodekey_map[qid] = quantize_nodekeys[qid.ia_op_exec_context.call_order]
            else:
                raise ValueError("Quantize Node not found for NWQid: {}".format(qid))

        if logger.level == logging.DEBUG:
            for qid, nodekey in qid_nodekey_map.items():
                logger.debug("QuantizerId: {}".format(qid))
                logger.debug("\tnodekey: {}".format(nodekey))

        return qid_nodekey_map


    def _dump_master_df(self):
        self.master_df.drop('state_module', axis=1).to_csv(
            osp.join(self.dump_dir, self.model_name + "_quantizable_state_table.csv"), index_label="nodestr")


    def _dump_quantized_graph(self):
        self.qmodel.get_graph().visualize_graph(osp.join(self.dump_dir, "qenv_graph.dot"))


    def _dump_groups_of_adjacent_quantizers(self):
        adj_quantizer_groups = []

        for i, _ in enumerate(self._groups_of_adjacent_quantizers):
            group_members = []
            for _, aq in enumerate(self._groups_of_adjacent_quantizers[i].activation_quantizers):
                group_members.append(self.master_df.index[self.master_df.qid == str(aq[0])][0])
            for _, wq in enumerate(self._groups_of_adjacent_quantizers[i].weight_quantizers):
                group_members.append(self.master_df.index[self.master_df.qid == str(wq[0])][0])
            adj_quantizer_groups.append(natsorted(group_members))

        with safe_open(self.dump_dir / "{}_groups_of_adjacent_quantizers.json".format(self.model_name), "w") as DUMP_FH:
            json.dump(natsorted(adj_quantizer_groups), DUMP_FH, indent=4)
