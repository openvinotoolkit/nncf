import logging

import os.path as osp
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import warnings

import json
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from natsort import natsorted
from collections import OrderedDict

from functools import partial
from shutil import copyfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10, CIFAR100

from nncf.nncf_logger import logger

from nncf import create_compressed_model
from nncf.compression_method_api import CompressionLevel
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.initialization import register_default_init_args
from sklearn.preprocessing import MinMaxScaler

from nncf.auto.utils.utils import AverageMeter, topk_accuracy, annotate_model_attr

from collections import OrderedDict, Counter
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId, InputQuantizerId, FunctionQuantizerId
from nncf.dynamic_graph.context import Scope
from natsort import natsorted

from nncf.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers, AdjacentQuantizers
from nncf.hw_config import HWConfigType
from nncf.quantization.precision_init.hawq_init import BitwidthAssignmentMode
from nncf.quantization.layers import BaseQuantizer

# logging
def prRed(prt): logger.info("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): logger.info("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): logger.info("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): logger.info("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): logger.info("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): logger.info("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): logger.info("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): logger.info("\033[98m {}\033[00m" .format(prt))

def find_qid_by_str(quantization_controller, qid_str):
    for _qid, _q in quantization_controller.all_quantizations.items():
        if qid_str == str(_qid):
            return _qid

class QuantizationEnv:
    def __init__(self, 
            quantization_controller, 
            eval_loader, 
            eval_fn,
            config: 'NNCFConfig'):

        logger.info("[Q.Env] Instantiating NNCF Quantization Environment")
        self.qctrl            = quantization_controller
        self.qmodel           = quantization_controller._model
        self.eval_loader      = eval_loader
        self.eval_fn          = eval_fn
        self.config           = config

        # TODO: Do we need to a way to specify a different device for automated search?
        # self.config.current_gpu = self.config.gpu_id
        # torch.cuda.set_device(config.gpu_id)         # Set default operating cuda device
        # self.config.device = get_device(self.config) # get_device requires config.current_gpu
        
        # Extract model name for labelling
        self.model_name = self.config.get('model', None)
        if self.model_name is None:
            self.model_name = self.pretrained_model.__class__.__name__

        # Check and only proceed if target device is supported by Q.Env
        self.hw_cfg_type = self.qctrl.quantization_config.get("hw_config_type")
        if self.hw_cfg_type is not None and self.hw_cfg_type is not HWConfigType.VPU:
            raise ValueError("Unsupported device ({}). Automatic Precision Initialization only supports for target_device NONE or VPU".format(self.hw_cfg_type.value))
        
        # Extract Precision Initialization Config
        if 'autoq' == self.config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {}):
            self.autoq_cfg = self.config.get('compression', {}).get('initializer', {}).get('precision')
        else:
            raise ValueError("Missing/Invalid Config of Precision Initializer. "
                             "Pls review config['compression']['initializer']['precision']")

        # Set target compression ratio
        self.compression_ratio = self.autoq_cfg.get('compression_ratio', 0.15)
        
        # Bool to disable hard resource constraint
        self.skip_wall = False
        if 'skip_wall' in self.autoq_cfg:
            self.skip_wall = self.autoq_cfg['skip_wall']

        # Bool to enable fine-tuning in each episode. Placeholder for now
        self.finetune = False
        if 'finetune' in self.autoq_cfg:
            self.finetune = self.autoq_cfg['finetune']

        # Configure search space for precision according to target device
        if self.hw_cfg_type is None:
            self.bitwidth_space = self.autoq_cfg.get('bits', [2, 4, 8])
            self.bw_assignment_mode = BitwidthAssignmentMode.LIBERAL
        elif self.hw_cfg_type is HWConfigType.VPU:
            self.bitwidth_space = self.qctrl._hw_precision_constraints.get_all_unique_bits()
            self.bw_assignment_mode = BitwidthAssignmentMode.STRICT
        self.bitwidth_space = sorted(list(self.bitwidth_space))
        self.float_bitwidth = 32.0
        self.max_bitwidth = max(self.bitwidth_space)
        self.min_bitwidth = min(self.bitwidth_space)
        
        # Quantizer Master Table Creation
        self._groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers(self.qctrl)
        self.quantizer_table = self._create_quantizer_table()

        # Create master dataframe to keep track of quantizable layers and thier attributes
        self.master_df, self.state_list = self._get_state_space(self.qctrl, self.qmodel, self.quantizer_table)
        if self.master_df.isnull().values.any():
            raise ValueError("Q.Env Master Dataframe has null value(s)")

        assert len(self.quantizer_table) == len(self.qctrl.all_quantizations), \
            "Number of Quantizer is not tally between quantizer table and quantization controller"
        
        # MinMaxScaler for State Embedding
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.master_df[self.state_list])

        # Model Size Calculation
        self.orig_model_size   = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.float_bitwidth  #in bit unit
        self.min_model_size    = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.min_bitwidth    # This variable has only been used once, just to ensure size constrainer doesnt go below than this
        self.target_model_size = self.orig_model_size*self.compression_ratio

        # Evaluate and store metric score of pretrained model
        self._evaluate_pretrained_model()

        # init reward
        self.best_reward = -math.inf #TODO: move reward to search manager
        self.reset()
        
        # Serialize Q.Env information. Note that these functions should be at the end of Q.Env Initialization.
        self._dump_master_df()
        self._dump_quantized_graph()
        self._dump_groups_of_adjacent_quantizers()

        # End of QuantizationEnv.__init__()
        # ----------------------------------------------------------------------------------------------------------------------


    def reset(self):
        self.collected_strategy=[]
        self.strategy=[]
        self.master_df['action']=max(self.bitwidth_space)
        self.master_df['prev_action']=0
        self.master_df['unconstrained_action']=0


    def _create_quantizer_table(self):
        # Create a mapping of qid to its adjacent quantizer group id
        adjq_gid_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())
        for qid, qmod in self.qctrl.all_quantizations.items():
            adjq_gid_map[qid] = self._groups_of_adjacent_quantizers.get_group_id_for_quantizer(qmod)

        # Create a mapping of qid to its bitwidth space
        bw_space_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())
        if self.hw_cfg_type is None:
            for qid in bw_space_map.keys():
                bw_space_map[qid] = self.bitwidth_space
        else:
            assert hasattr(self.qctrl._hw_precision_constraints, '_constraints'), \
                "feasible bitwidth per quantizer not found"

            for qid, bw_space in self.qctrl._hw_precision_constraints._constraints.items():
                bw_space_map[qid] = sorted(list(bw_space))

        assert len(set(bw_space_map.keys()) - set(adjq_gid_map.keys())) == 0, \
            "both bw_space_map and adjq_gid_map must have exact keys."
        
        # Create a mapping of qid to its nodekey in NNCFGraph 
        qid_nodekey_map = self._generate_qid_nodekey_map(self.qctrl, self.qmodel)

        assert len(set(qid_nodekey_map.keys()) - set(adjq_gid_map.keys())) == 0, \
            "both qid_nodekey_map and adjq_gid_map must have exact keys."

        d = OrderedDict()
        for qid, qmod in self.qctrl.all_quantizations.items():
            nodekey     = qid_nodekey_map[qid]
            q_nx_nodeid = nodekey.split()[0]
            idx_str     = '-'.join([q_nx_nodeid, str(qid)])
            gid         = adjq_gid_map[qid]

            d[idx_str]                 = OrderedDict()
            d[idx_str]['qid']          = str(qid)
            d[idx_str]['q_nx_nodeid']  = q_nx_nodeid
            d[idx_str]['q_nx_nodekey'] = nodekey
            d[idx_str]['gid']          = gid
            d[idx_str]['bw_space']     = bw_space_map[qid]
            d[idx_str]['is_pred']      = True
            
            if isinstance(qid, WeightQuantizerId):
                d[idx_str]['state_scope'] = qid.scope
            elif isinstance(qid, NonWeightQuantizerId):
                d[idx_str]['state_scope'] = qid.ia_op_exec_context.scope_in_model
            else:
                raise NotImplementedError("QuantizerId: {} of {} class is not supported.".format(str(qid, qid.__class__.__name__)))

        # quantizer_table index is QuantizerId in string prepended with its quantize node id in NNCFGraph
        df                    = pd.DataFrame.from_dict(d, orient='index')
        df['qid_obj']         = df['qid'].apply(lambda x: find_qid_by_str(self.qctrl, x))
        df['qmodule']         = df['qid_obj'].apply(lambda x: self.qctrl.all_quantizations[x])
        df['is_wt_quantizer'] = df['qmodule'].apply(lambda x: x.is_weights)
        df['state_module']    = df['state_scope'].apply(lambda x: self.qmodel.get_module_by_scope(x))

        quantizer_table = df.loc[natsorted(df.index)]
        return quantizer_table


    def _get_state_space(self, quantization_controller, quantized_model, quantizer_table):
        def annotate_learnable_module_io_shape(model):
            def annotate_io_shape(module, input_, output):
                if hasattr(module, 'weight') or isinstance(module, BaseQuantizer):
                    module._input_shape  = input_[0].shape
                    module._output_shape = output.shape

            hook_list = [m.register_forward_hook(annotate_io_shape) for n, m in model.named_modules()]
            model.do_dummy_forward(force_eval=True)
            for h in hook_list:
                h.remove()
            
        annotate_learnable_module_io_shape(quantized_model)

        # State Embedding Extraction
        #---------------------------
        df = quantizer_table
        layer_attr_df                     = df.apply(self._get_layer_attr, axis=1)
        layer_attr_df['layer_idx']        = np.array(range(len(layer_attr_df)))
        layer_attr_df['weight_quantizer'] = df['is_wt_quantizer'].astype('float')
        state_list = layer_attr_df.columns.to_list()
       
        # create master dataframe
        master_df = pd.concat([df, layer_attr_df], axis='columns')
        
        # TODO: Revision Needed. Workaround to set the min and max of action before fitting the minmaxscaler
        master_df['prev_action'][0]=self.max_bitwidth
        master_df['prev_action'][-1]=self.min_bitwidth

        return master_df, state_list
    
    
    def _get_layer_attr(self, row):        
        g       = self.qmodel.get_graph()
        m       = row.state_module
        qid     = row.qid_obj
        feature = OrderedDict()

        if isinstance(qid, WeightQuantizerId):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                feature['conv_dw']          = int(m.weight.shape[1] == m.groups) # 1.0 for depthwise, 0.0 for other conv2d
                feature['cin']              = m.weight.shape[1]
                feature['cout']             = m.weight.shape[0]
                feature['stride']           = m.stride[0]
                feature['kernel']           = m.kernel_size[0]
                feature['param']            = np.prod(m.weight.size())     
                feature['ifm_size']         = np.prod(m._input_shape[-2:]) # H*W
                feature['prev_action']      = 0.0 # placeholder                

            elif isinstance(m, nn.Linear):
                feature['conv_dw']          = 0.0 
                feature['cin']              = m.in_features
                feature['cout']             = m.out_features
                feature['stride']           = 0.0
                feature['kernel']           = 1.0
                feature['param']            = np.prod(m.weight.size())     
                feature['ifm_size']         = np.prod(m._input_shape[-1]) # feature nodes
                feature['prev_action']      = 0.0 # placeholder  
            
            else:
                raise NotImplementedError("State embedding extraction of {}".format(m.__class__.__name__))

        elif isinstance(qid, NonWeightQuantizerId):
            qmod = self.qctrl.all_quantizations[qid]
            input_shape  = qmod._input_shape
            output_shape = qmod._output_shape
            feature['cin']         = input_shape[1] if len(input_shape) == 4 else input_shape[-1]
            feature['cout']        = output_shape[1] if len(output_shape) == 4 else output_shape[-1]
            feature['ifm_size']    = np.prod(input_shape[-2:]) if len(input_shape) == 4 else input_shape[-1]
            feature['conv_dw']     = 0.0 
            feature['stride']      = 0.0
            feature['kernel']      = 0.0
            feature['param']       = 0.0     
            feature['prev_action'] = 0.0 

            if len(input_shape) != 4 and len(input_shape) != 2:
                raise NotImplementedError("A design is required to cater this scenario. Pls. report to maintainer")

        elif isinstance(qid, InputQuantizerId):
            raise NotImplementedError("InputQuantizerId is supported, quantizer of nncf model input is of type NonWeightQuantizerId")
        
        elif isinstance(qid, FunctionQuantizerId):
            raise NotImplementedError("FunctionQuantizerId is supported, Pls. report to maintainer")
        
        else:
            raise ValueError("qid is an instance of unexpected class {}".format(qid.__class__.__name__))
        
        return pd.Series(feature)


    def _evaluate_pretrained_model(self):
        # Registered evaluation function is expected to return a single scalar score
        logger.info("[Q.Env] Evaluating Pretrained Model")
        self.qctrl.disable_weight_quantization()
        self.qctrl.disable_activation_quantization()

        with torch.no_grad():
            self.pretrained_score = self.eval_fn(self.qmodel, self.eval_loader)
            logger.info("Pretrained Score: {:.2f}".format(self.pretrained_score))
        
        self.qctrl.enable_weight_quantization()
        self.qctrl.enable_activation_quantization()
        self.qmodel.rebuild_graph()


    def _adaptbn(self):
        train_mode = self.qmodel.training
        if not train_mode:
            self.qmodel.train()

        self.qctrl.run_batchnorm_adaptation(self.qctrl.quantization_config)
        
        if not train_mode:
            self.qmodel.eval()

       
    def _run_quantization_pipeline(self, finetune):
        self._adaptbn()
        if finetune:
            raise NotImplementedError("Post-Quantization fine tuning is not implemented.")
        else:
            with torch.no_grad():
                quantized_score = self.eval_fn(self.qmodel, self.eval_loader)
                logger.info("[Q.Env] Post-Init: {:.3f}".format(quantized_score))
        return quantized_score


    def _calc_quantized_model_size(self): # in bit
        assert len(self.strategy) == len(self.master_df) # This function is only allowed when all actions are predicted        
        return sum(self.master_df['param'] * self.master_df.is_wt_quantizer * self.master_df['action'])


    def _expand_collected_strategy(self, collected_strategy):
        grouped_strategy_map = OrderedDict(zip(list(self.master_df.index[self.master_df.is_pred]), collected_strategy))
        for nodestr, action in grouped_strategy_map.items():
            self.master_df.loc[nodestr, 'action']=action # This is needed for dangling quantizer

            # Extract list of qid of adjacent quantizer in the group, then apply same action to them in the master_df
            group_id = self._groups_of_adjacent_quantizers.get_group_id_for_quantizer(self.master_df.qmodule[nodestr])
            qid_in_group = list(map(lambda qid_qmod_pair: qid_qmod_pair[0], 
                                    self._groups_of_adjacent_quantizers.get_adjacent_quantizers_by_group_id(group_id)))
            for qid in qid_in_group:
                self.master_df.loc[self.master_df.qid == str(qid), 'action'] = action

        return list(self.master_df['action'])


    def _final_action_wall(self, skip=False):
        # This function acts on self.strategy and return self.strategy
        def lower_precision(precision):
            d = {8:4, 4:2, 2:2}
            return d[precision]

        if skip is not True:
            self.master_df['unconstrained_action']=self.master_df['action']
            cur_model_size = self._calc_quantized_model_size()
            while self.min_model_size < cur_model_size and self.target_model_size < cur_model_size:
                
                # Tricky part TODO need to propagate for any precision that has been pushed lower
                for i, nodestr in enumerate( reversed(self.master_df.index.tolist()) ):
                    
                    # if self.master_df.loc[nodestr, "nparam"] > 0: #input quantizer has nparam > 0
                    if self.master_df.loc[nodestr, "is_wt_quantizer"] & self.master_df.loc[nodestr, "is_pred"]: 
                        n_bit = self.master_df.loc[nodestr, 'action']
                        new_bit = lower_precision(n_bit)
                        if new_bit != n_bit:
                            self.master_df.loc[nodestr, "action"] = new_bit

                            # Extract list of qid of adjacent quantizer in the group, then apply same action to them in the master_df
                            group_id = self._groups_of_adjacent_quantizers.get_group_id_for_quantizer(self.master_df.qmodule[nodestr])
                            qid_in_group = list(map(lambda qid_qmod_pair: qid_qmod_pair[0],
                                                    self._groups_of_adjacent_quantizers.get_adjacent_quantizers_by_group_id(group_id)))
                            for qid in qid_in_group:
                                self.master_df.loc[self.master_df.qid == str(qid), 'action'] = new_bit

                    #strategy update here
                    self.strategy = self.master_df['action']
                    cur_model_size = self._calc_quantized_model_size()
                    if cur_model_size <= self.target_model_size:
                        break
        else:
            print("=> Skip action constraint")

        # TODO This whole section has to be revised
        self.strategy = self.master_df['action']
        logger.info('=> Final action list: {}'.format(self.strategy.astype('int').to_list()))

        for i, nodestr in enumerate(self.master_df.index.tolist()):
            if self.master_df.loc[nodestr, 'action'] == self.master_df.loc[nodestr, 'unconstrained_action']:
                logger.info("Precision[{:>4}]: {} | {}".format(i, self.master_df.loc[nodestr, 'action'], str(nodestr)))
            else:
                logger.info("Precision[{:>4}]: {} <= {} | {}".format(i, self.master_df.loc[nodestr, 'action'], self.master_df.loc[nodestr, 'unconstrained_action'], str(nodestr)))

        return self.strategy

    def reward(self, acc, model_ratio):
        return (acc - self.pretrained_score) * 0.1 

    def step(self, action):
        def is_final_step():
            if self.bw_assignment_mode is BitwidthAssignmentMode.STRICT:
                return len(self.collected_strategy) == sum(self.master_df.is_pred)
            else:
                return len(self.collected_strategy) == len(self.master_df)
        
        self.collected_strategy.append(action)  # save action to strategy

        if not is_final_step():
            info_set = {}
            reward = 0
            is_strict = self.bw_assignment_mode is BitwidthAssignmentMode.STRICT
            self.set_next_step_prev_action(len(self.collected_strategy), action, only_pred=is_strict)
            obs = self.get_normalized_obs(len(self.collected_strategy), only_pred=is_strict)
            done = False
            return obs, reward, done, info_set

        else:
            return self.evaluate_strategy(self.collected_strategy, skip_wall=self.skip_wall)
        
    def evaluate_strategy(self, collected_strategy, skip_wall=True):
        # #Expand strategy to full quantization policy
        if self.bw_assignment_mode is BitwidthAssignmentMode.STRICT:
            self.strategy = self._expand_collected_strategy(collected_strategy)
        else:
            self.master_df['action'] = collected_strategy
            self.strategy = self.master_df['action']
        
        if skip_wall is not True:
            self.strategy = self._final_action_wall()
        
        assert len(self.strategy) == len(self.master_df)

        # Quantization
        self.apply_actions(self.strategy)
        
        cur_model_size = self._calc_quantized_model_size()
        cur_model_ratio = cur_model_size / self.orig_model_size

        for idx, qid in zip(self.master_df.index, self.master_df['qid']):
            logger.info("[Q.Env] {:50} | {}".format(
                str(self.qctrl.all_quantizations[ find_qid_by_str(self.qctrl, qid) ]),
                idx))

        quantized_score = self._run_quantization_pipeline(finetune=self.finetune)
        reward = self.reward(quantized_score, cur_model_ratio)
        
        info_set = {'model_ratio': cur_model_ratio, 'accuracy': quantized_score, 'model_size': cur_model_size}

        if reward > self.best_reward:
            self.best_reward = reward
            prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, model_ratio: {:.3f}, model_size(mb): {:.3f}'.format(
                self.strategy, self.best_reward, quantized_score, cur_model_ratio, cur_model_size/8000000))

        obs = self.get_normalized_obs(len(collected_strategy)-1, 
                                        only_pred=(self.bw_assignment_mode is BitwidthAssignmentMode.STRICT))
                                                    
        done = True
        return obs, reward, done, info_set

    def set_next_step_prev_action(self, idx, action, only_pred=True):
        if only_pred:
            self.master_df.loc[self.master_df.index[self.master_df.is_pred][idx], 'prev_action'] = action
        else:
            self.master_df.loc[self.master_df.index[idx], 'prev_action'] = action

    def get_normalized_obs(self, idx, only_pred=True):
        if only_pred:
            _df = self.master_df.loc[self.master_df.is_pred, self.state_list]
        else:
            _df = self.master_df.loc[self.master_df.index, self.state_list]
        _df.loc[_df.index, self.state_list] =  self.state_scaler.transform(_df[self.state_list])
        return _df.iloc[idx]

    def apply_actions(self, strategy):
        self.master_df['action']=self.strategy

        # step_cfg = deepcopy(self.orig_compress_cfg)
        step_cfg = self.config['compression']

        if 'scope_overrides' not in step_cfg:
            step_cfg['scope_overrides']={}

        if 'ignored_scopes' not in step_cfg:
            step_cfg['ignored_scopes']=[]

        # TODO: Handling for keeping layer at FP32
        for layer in self.master_df.index:
            precision=self.master_df.loc[layer, "action"]

            if self.master_df.loc[layer, "is_wt_quantizer"]:
                ScopeStr = str(self.master_df.loc[layer, 'state_scope'])
                step_cfg['scope_overrides'][ScopeStr] = {}
                step_cfg['scope_overrides'][ScopeStr]['bits']=int(precision) # int requires to convert numpy.int64 to int
            else:
                # if we use scope to non-weight quantizer, we would risk masking out the quantizers within the scope
                # e.g. MobileNetV2/adaptive_avg_pool2d_0 masks all quantizers into the same scope
                IAOpCtxStr = str(self.master_df.loc[layer, 'qid_obj'].ia_op_exec_context)
                step_cfg['scope_overrides'][IAOpCtxStr] = {}
                step_cfg['scope_overrides'][IAOpCtxStr]['bits']=int(precision)

            self.master_df.loc[layer, "qmodule"].num_bits = precision # Actual actor to change quantizer precision
        self.config['compression']=step_cfg
        return True 


    def _generate_qid_nodekey_map(self, quantization_controller: 'QuantizationController', quantized_network: 'NNCFNetwork'):
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

        g=quantized_network.get_graph()

        for nodekey in g.get_all_node_keys():
            if 'symmetric_quantize' in nodekey and 'UpdateWeight' in nodekey:
                weight_quantize_nodekeys.append(nodekey)
            if 'symmetric_quantize' in nodekey and 'UpdateWeight' not in nodekey:
                non_weight_quantize_nodekeys.append(nodekey)

        # Find nodekey of Weight Quantizer
        for qid, qmod in quantization_controller.weight_quantizers.items():
            quantize_nodekeys = []
            for nodekey in weight_quantize_nodekeys:
                if str(qid.scope) in nodekey:
                    quantize_nodekeys.append(nodekey)

            if len(quantize_nodekeys) == 1:
                qid_nodekey_map[qid]=quantize_nodekeys[0]
            else:
                raise ValueError("Quantize Node not found or More Nodes are found for WQid: {}".format(qid))

        # Find nodekey of Non-Weight Quantizer
        for qid, qmod in quantization_controller.non_weight_quantizers.items():
            quantize_nodekeys = []
            for nodekey in non_weight_quantize_nodekeys:
                if str(qid.ia_op_exec_context.scope_in_model) in nodekey:
                    quantize_nodekeys.append(nodekey)

            if len(quantize_nodekeys) > 0:
                qid_nodekey_map[qid]=quantize_nodekeys[qid.ia_op_exec_context.call_order]
            else:
                raise ValueError("Quantize Node not found for NWQid: {}".format(qid))
        
        if logger.level == logging.DEBUG:
            for qid, nodekey in qid_nodekey_map.items():
                logger.debug("QuantizerId: {}".format(qid))
                logger.debug("\tnodekey: {}".format(nodekey))
        
        return qid_nodekey_map


    def _dump_master_df(self):
        self.master_df.drop('state_module', axis=1).to_csv(
            osp.join(self.config['log_dir'], self.model_name + "_quantizable_state_table.csv"), index_label="nodestr")


    def _dump_quantized_graph(self):
        self.qmodel.get_graph().visualize_graph(osp.join(self.config.get("log_dir", "."), "qenv_graph.dot"))


    def _dump_groups_of_adjacent_quantizers(self):
        adj_quantizer_groups = []

        for i, g in enumerate(self._groups_of_adjacent_quantizers):
            group_members = []
            for j, aq in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].activation_quantizers):
                group_members.append(self.master_df.index[self.master_df.qid == str(aq[0])][0])
            for k, wq in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers):
                group_members.append(self.master_df.index[self.master_df.qid == str(wq[0])][0])
            adj_quantizer_groups.append(natsorted(group_members))

        with open(osp.join(self.config.get("log_dir", "."), 
                            self.model_name + "_groups_of_adjacent_quantizers.json"), "w") as DUMP_FH:
            json.dump(natsorted(adj_quantizer_groups), DUMP_FH, indent=4)


    def _get_learnable_qids_and_leads(self):
        # By default, all weight quantizers must be learnable to allow optimal model size compression
        # Bitwidth assignment mode only applies to activation quantizers of a group. 
        # When the mode is LIBERAL, all activation quantizers are learnable.
        # When it is STRICT, none of the activation quantizers is learnable but to follow the leader of the group.
        # HW config determines the bitwidth space of a quantizer

        learnable_qids = []
        lead_qid_of_groups = [None]*len(list(self._groups_of_adjacent_quantizers))

        if self.bw_assignment_mode is BitwidthAssignmentMode.STRICT:

            for i, g in enumerate(self._groups_of_adjacent_quantizers):
                n_wq = len(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers)
                n_aq = len(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].activation_quantizers)

                if n_wq > 0:
                    # By default, all weight quantizers are learnable when it is unconstrained by HW. 
                    # If target HW is specified, a weight quantizer is only learnable if mixed-precision support is available for its associated weight layer
                    for k, (wqid, wqmod) in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers):
                        if self.hw_cfg_type is None or len(self.qctrl._hw_precision_constraints._constraints[wqid]) > 1:
                            learnable_qids.append(wqid)
                        
                    # We will assume the first weight quantizer as the leader of the group, 
                    # Leader means its precision decides for the rest of activation quantizers
                    for k, (wqid, wqmod) in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers):
                        if self.hw_cfg_type is None or len(self.qctrl._hw_precision_constraints._constraints[wqid]) > 1:
                            lead_qid_of_groups[i] = self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers[k][0]
                            break
                else:
                    # when there is no weight quantizer in the group, the first activation quantizer
                    # is the leader of the group and learnable if mixed-precision is supported
                    for j, (aqid, aqmod) in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].activation_quantizers):
                        if self.hw_cfg_type is None or len(self.qctrl._hw_precision_constraints._constraints[aqid]) > 1:
                            learnable_qids.append(aqid)
                            lead_qid_of_groups[i] = self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].activation_quantizers[j][0]
                            break
                        
        if self.bw_assignment_mode is BitwidthAssignmentMode.LIBERAL:
            if self.hw_cfg_type is None:
                learnable_qids += list(self.qctrl.all_quantizations.keys())
            else: # VPU device
                # quantizers will only be added to learnable list if mixed-precision support available for its associated operator.
                for k, (wqid, wqmod) in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[i].weight_quantizers):
                    if len(self.qctrl._hw_precision_constraints._constraints[wqid]) > 1:
                        learnable_qids.append(wqid)
                for aqid, aqmod in self.qctrl.activation_quantizers.items():
                    if len(self.qctrl._hw_precision_constraints._constraints[aqid]) > 1:
                        learnable_qids.append(aqid)

        return learnable_qids, lead_qid_of_groups




