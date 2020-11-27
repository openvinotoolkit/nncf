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

# logging
def prRed(prt): logger.info("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): logger.info("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): logger.info("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): logger.info("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): logger.info("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): logger.info("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): logger.info("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): logger.info("\033[98m {}\033[00m" .format(prt))

def find_qidobj(quantization_controller, qid_str):
    for _qid, _q in quantization_controller.all_quantizations.items():
        if qid_str == str(_qid):
            return _qid

class QuantizationEnv:
    def __init__(self, 
            quantization_controller, 
            criterion, 
            train_loader, 
            val_loader,
            train_epoch_fn,
            validate_fn,
            config: 'NNCFConfig'):

        logger.info("[Q.Env] Instantiating NNCF Quantization Environment")
        self.qctrl            = quantization_controller
        self.qmodel           = quantization_controller._model
        self.criterion        = criterion
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.train_epoch_fn   = train_epoch_fn
        self.validate_fn      = validate_fn
        self.config           = config

        # TODO: Do we need to a way to specify a different device for automated search?
        # self.config.current_gpu = self.config.gpu_id
        # torch.cuda.set_device(config.gpu_id)         # Set default operating cuda device
        # self.config.device = get_device(self.config) # get_device requires config.current_gpu
        
        # Model label
        self.model_name = self.config.get('model', None)
        if self.model_name is None:
            self.model_name = self.pretrained_model.__class__.__name__

        self.hw_cfg_type = self.qctrl.quantization_config.get("hw_config_type")
        if self.hw_cfg_type is not None and self.hw_cfg_type is not HWConfigType.from_str('VPU'):
            raise ValueError("Unsupported device ({}). Automatic Precision Initialization only supports for target_device NONE or VPU".format(self.hw_cfg_type.value))
        
        if 'autoq' == self.config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {}):
            self.autoq_cfg = self.config.get('compression', {}).get('initializer', {}).get('precision')
        else:
            raise ValueError("Missing/Invalid Config of Precision Initializer. "
                             "Pls review config['compression']['initializer']['precision']")

        # Target Compression Ratio
        self.compression_ratio = self.autoq_cfg.get('compression_ratio', 0.15)
        
        # Set Precision Space and Adjacent Quantizer Coupling
        if self.hw_cfg_type is None:
            self.precision_space = self.autoq_cfg.get('bits', [4, 8])
            self.tie_quantizers = False
        elif self.hw_cfg_type is HWConfigType.from_str('VPU'):
            self.precision_space = self.qctrl._hw_precision_constraints.get_all_unique_bits()
            self.tie_quantizers = True
        self.precision_space = sorted(self.precision_space)
        self.float_bit = 32.0
        self.max_bit = max(self.precision_space)
        self.min_bit = min(self.precision_space)
        
        # Bool to disable hard resource constraint
        self.skip_wall = False
        if 'skip_wall' in self.autoq_cfg:
            self.tie_quantizers = self.autoq_cfg['skip_wall']

        # Bool to enable fine-tuning in each episode. Placeholder for now
        self.finetune = False
        if 'finetune' in self.autoq_cfg:
            self.finetune = self.autoq_cfg['finetune']

        # TODO (Design): How to generalize evaluation and train in general? tasks could have different metric and loss function
        self._evaluate_pretrained_model()

        # Quantizer Master Table Creation
        self._groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers(self.qctrl)
        qid_nodekey_map = self._generate_qid_nodekey_map(self.qctrl, self.qmodel)
        d = OrderedDict()
        for gid, group in enumerate(self._groups_of_adjacent_quantizers):           
            for aq_id, aq in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[gid].activation_quantizers):
                qid=aq[0]
                nodekey=qid_nodekey_map[qid]
                q_nx_nodeid=nodekey.split()[0]
                idx_str = '-'.join([q_nx_nodeid, str(qid)])

                d[idx_str] = OrderedDict()
                d[idx_str]['qid'] = str(qid)
                d[idx_str]['q_nx_nodeid'] = q_nx_nodeid
                d[idx_str]['q_nx_nodekey'] = nodekey
                d[idx_str]['state_scope'] = qid.ia_op_exec_context.scope_in_model
                d[idx_str]['gemm_nx_nodekey'] = list(map(lambda x: qid_nodekey_map[x[0]], 
                                                    self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[gid].weight_quantizers))

            for wq_id, wq in enumerate(self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[gid].weight_quantizers):
                qid=wq[0]
                nodekey=qid_nodekey_map[qid]
                q_nx_nodeid=nodekey.split()[0]
                idx_str = '-'.join([q_nx_nodeid, str(qid)])

                d[idx_str] = OrderedDict()
                d[idx_str]['qid'] = str(qid)
                d[idx_str]['q_nx_nodeid']  = q_nx_nodeid
                d[idx_str]['q_nx_nodekey'] = nodekey
                d[idx_str]['state_scope'] = qid.scope
                d[idx_str]['gemm_nx_nodekey'] = list(map(lambda x: qid_nodekey_map[x[0]], 
                                                    self._groups_of_adjacent_quantizers._groups_of_adjacent_quantizers[gid].weight_quantizers))

        # qtable index is QID in string prepended with its quantize node id
        df = pd.DataFrame.from_dict(d,orient='index')
        qtable = df.loc[natsorted(df.index)]

        # # Consolidate a flag per quantizer to signify if the precision should be learned
        # qtable['is_pred'] = True # Assume NONE device, precision of all quantizers will be learned
        # if self.tie_quantizers is True: # VPU device will enter the loop
        #     qtable['is_pred'] = False

        #     # TODO
        #     # We loop through GroupsOfAdjacentQuantizers
        #     # For each group, if there is fixed bitwidth in any of adjacent quantizer, no precision learning for the group
        #     # Otherwise, a single precision will need to be learned for the group.
        #     # Specifically where weight quantizer will be learned, 
        #     # other adjacent quantizers to follow. If W quantizer does not exist in a group, we choose by node id order.

        # Create master dataframe to keep track of quantizable layers and thier attributes, a.k.a state embedding
        self.master_df, self.state_list = self._get_state_space(self.qctrl, self.qmodel, qtable)
    
        # MinMaxScaler for State Embedding
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.master_df[self.state_list])

        # Model Size Calculation
        self.orig_model_size   = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.float_bit  #in bit unit
        self.min_model_size    = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.min_bit    # This variable has only been used once, just to ensure size constrainer doesnt go below than this
        self.target_model_size = self.orig_model_size*self.compression_ratio

        # init reward
        self.best_reward = -math.inf #TODO: move reward to search manager
        self.reset()
        
        # Serialize Q.Env information. Note that these functions should be at the end of Q.Env Initialization.
        self._dump_master_df()         
        self._dump_groups_of_adjacent_quantizers()
        self._dump_quantized_graph()

        # End of QuantizationEnv.__init__()
        # ----------------------------------------------------------------------------------------------------------------------

    def reset(self):
        self.collected_strategy=[]
        self.strategy=[]
        self.master_df['action']=0
        self.master_df['prev_action']=0
        self.master_df['unconstrained_action']=0

    def _evaluate_pretrained_model(self):
        # Registered evaluation function is expected to return a single scalar score
        logger.info("[Q.Env] Evaluating Pretrained Model")
        self.qctrl.disable_weight_quantization()
        self.qctrl.disable_activation_quantization()

        with torch.no_grad():
            self.pretrained_score = self.validate_fn(self.val_loader, self.qmodel, self.criterion, self.config)
            logger.info("Pretrained Score: {:.2f}".format(self.pretrained_score))
        
        self.qctrl.enable_weight_quantization()
        self.qctrl.enable_activation_quantization()
        self.qmodel.rebuild_graph()

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
            if self.tie_quantizers is True:
                return len(self.collected_strategy) == sum(self.master_df.is_pred)
            else:
                return len(self.collected_strategy) == len(self.master_df)
        
        self.collected_strategy.append(action)  # save action to strategy

        if not is_final_step():
            info_set = {}
            reward = 0
            
            self.set_next_step_prev_action(len(self.collected_strategy), action, only_pred=self.tie_quantizers)
            obs = self.get_normalized_obs(len(self.collected_strategy), only_pred=self.tie_quantizers)
            done = False
            return obs, reward, done, info_set

        else:
            return self.evaluate_strategy(self.collected_strategy, skip_wall=self.skip_wall)
        
    def evaluate_strategy(self, collected_strategy, skip_wall=True):
        # #Expand strategy to full quantization policy
        if self.tie_quantizers:
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
                str(self.qctrl.all_quantizations[ find_qidobj(self.qctrl, qid) ]),
                idx))

        quantized_acc = self._run_quantization_pipeline(finetune=self.finetune)
        reward = self.reward(quantized_acc, cur_model_ratio)
        
        info_set = {'model_ratio': cur_model_ratio, 'accuracy': quantized_acc, 'model_size': cur_model_size}

        if reward > self.best_reward:
            self.best_reward = reward
            prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, model_ratio: {:.3f}, model_size(mb): {:.3f}'.format(
                self.strategy, self.best_reward, quantized_acc, cur_model_ratio, cur_model_size/8000000))

        obs = self.get_normalized_obs(len(collected_strategy)-1, only_pred=self.tie_quantizers)            
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
        
    def _get_state_space(self, quantization_controller, quantized_model, qtable):
        # TODO: can we use nncf utility to the following? like dummy forward?
        input_size = self.config['input_info']['sample_size']
        annotate_model_attr(quantized_model, tuple(input_size[1:])) # assume axis 0 be batch size

        df = qtable
        df['qid_obj']         = df['qid'].apply(lambda x: find_qidobj(quantization_controller, x))
        df['qmodule']         = df['qid_obj'].apply(lambda x: quantization_controller.all_quantizations[x])
        df['is_wt_quantizer'] = df['qmodule'].apply(lambda x: x.is_weights)
        df['state_module']    = df['state_scope'].apply(lambda x: quantized_model.get_module_by_scope(x))

        # getting the quantizers that we need agent to predict
        weight_quantizer_indices = df.index[df.is_wt_quantizer] # All weight quantizer precision will be predicted
        assert len(set(weight_quantizer_indices)) == len(weight_quantizer_indices), "master table cannot have duplicated row for same weight quantizer"
        
        remove_indices = []
        for associated_gemm_nx_nodekeys in df.gemm_nx_nodekey[df.gemm_nx_nodekey.apply(lambda x: len(x) > 1)]:
            followers = natsorted(associated_gemm_nx_nodekeys)[1:] #
            for _follower in followers:
                for idx, matcher in df.gemm_nx_nodekey[df.gemm_nx_nodekey.apply(lambda x: len(x) == 1)].items():
                    if _follower in matcher:
                        remove_indices.append(idx)
        assert len(set(remove_indices)) == len(remove_indices), "Why are there duplicates in remove_indices?"
                
        dangling_quantizer_indices = df.index[df.gemm_nx_nodekey.apply(lambda x: len(x) == 0)]
        assert len(set(dangling_quantizer_indices)) == len(dangling_quantizer_indices), "master table cannot have duplicated row for same quantizer"

        consolidated_weight_quantizer_indices = list(set(weight_quantizer_indices)-set(remove_indices))

        final_quantizable_indices = pd.Index(natsorted(list(set(consolidated_weight_quantizer_indices).union(set(dangling_quantizer_indices)))))

        assert len(final_quantizable_indices) == len(consolidated_weight_quantizer_indices) + len(dangling_quantizer_indices), "length should be tally"

        if self.tie_quantizers is False:
            df['is_pred']=True
        else:
            df['is_pred']=False
            df.loc[final_quantizable_indices, 'is_pred']=True
        
        # State Embedding
        #----------------
        layer_attr_df                     = df.apply(self._get_layer_attr, axis=1)
        layer_attr_df['layer_idx']        = np.array(range(len(layer_attr_df)))
        layer_attr_df['weight_quantizer'] = df['is_wt_quantizer'].astype('float')
        state_list = layer_attr_df.columns.to_list()
       
        # create master dataframe
        master_df = pd.concat([df, layer_attr_df], axis='columns')
        
        # Workaround to set the min and max of action before fitting the minmaxscaler
        master_df['prev_action'][0]=self.max_bit
        master_df['prev_action'][-1]=self.min_bit

        return master_df, state_list
    
    
    def _get_layer_attr(self, row):        
        m = row['state_module']
        state_list=[]
        feature=OrderedDict()

        if isinstance(m, nn.Conv2d):
            feature['conv_dw']          = int(m.in_channels == m.groups) # 1.0 for depthwise, 0.0 for other conv2d
            feature['cin']              = m.in_channels
            feature['cout']             = m.out_channels
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
            assert len(m._input_shape) in [2,4] , "new condition encountered, pls revise design"

            if row['qid'] == 'MobileNetV2/adaptive_avg_pool2d_0':
                qm = row['qmodule']
                if len(m._input_shape) == 2:
                    feature['cin']              = qm._input_shape[-1]
                    feature['cout']             = qm._output_shape[-1]
                    feature['ifm_size']         = qm._input_shape[-1]
                elif len(m._input_shape) == 4:
                    feature['cin']              = qm._input_shape[1]
                    feature['cout']             = qm._output_shape[1]
                    feature['ifm_size']         = np.prod(qm._input_shape[-2:]) # H*W
            else:
                if len(m._input_shape) == 2:
                    feature['cin']              = m._input_shape[-1]
                    feature['cout']             = m._output_shape[-1]
                    feature['ifm_size']         = m._input_shape[-1]
                elif len(m._input_shape) == 4:
                    feature['cin']              = m._input_shape[1]
                    feature['cout']             = m._output_shape[1]
                    feature['ifm_size']         = np.prod(m._input_shape[-2:]) # H*W

            feature['conv_dw']          = 0.0 
            feature['stride']           = 0.0
            feature['kernel']           = 0.0 # what should we set for non-learnable layer?
            feature['param']            = 0.0     
            feature['prev_action']      = 0.0 # placeholder 
           
        return pd.Series(feature)
       
    def _run_quantization_pipeline(self, finetune):
        self._adaptbn()
        
        if finetune:
            raise NotImplementedError("Post-Quantization fine tuning is not implemented.")
        else:
            with torch.no_grad():
                quantized_acc = self.validate_fn(self.val_loader, self.qmodel, self.criterion, self.config)
                logger.info("[Q.Env] Post-Init: {:.3f}".format(quantized_acc))

        return quantized_acc

    def _adaptbn(self):
        train_mode = self.qmodel.training
        if not train_mode:
            self.qmodel.train()

        self.qctrl.run_batchnorm_adaptation(self.qctrl.quantization_config)
        
        if not train_mode:
            self.qmodel.eval()

    @staticmethod
    def _generate_qid_nodekey_map(quantization_controller: 'QuantizationController', quantized_network: 'NNCFNetwork'):
        """
        Create a lookup mapping for each QuantizerId to its corresponding quantize node in network graph
        :param quantization_controller: 
        :param quantized_network:
        :return: dict with key of QuantizerId and value of node key string
        """    
        # Map Non Weight Qid to its nodes in nxgraph
        weight_quantize_nodekeys = []
        non_weight_quantize_nodekeys = []
        qid_nodekey_map = dict()

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





