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

from examples.common.sample_config import SampleConfig
from examples.common.execution import ExecutionMode, get_device, prepare_model_for_execution
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.utils import is_pretrained_model_requested
from examples.common.example_logger import logger

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

from nncf.auto.environment.quantizer_tracing import \
    get_gemm_with_input_quantizers, get_untagged_quantizer

from nncf.auto.environment.quantizer_tracing import QuantizerTracer, find_qidobj

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

class QuantizationEnv:
    def __init__(self, 
            quantization_controller, 
            criterion, 
            train_loader, 
            val_loader,
            train_epoch_fn,
            validate_fn,
            config: SampleConfig):

        logger.info("[Q.Env] Instantiating NNCF Quantization Environment")
        self.qctrl            = quantization_controller
        self.qmodel           = quantization_controller._model
        self.criterion        = criterion
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.train_epoch_fn   = train_epoch_fn
        self.validate_fn      = validate_fn

        self.config = config
        if self.config.get('nncf_config', None) is None:
            raise KeyError("config must have nncf_config dictionary")
        
        # TODO: Do we need to a way to specify a different device for automated search?
        # self.config.current_gpu = self.config.gpu_id
        # torch.cuda.set_device(config.gpu_id)         # Set default operating cuda device
        # self.config.device = get_device(self.config) # get_device requires config.current_gpu
        
        # pretrained_model is assumed to have checkpoint loaded
        # if isinstance(pretrained_model, nn.Module):
        #     pretrained_model.to(self.config.device)
        #     self.pretrained_model = pretrained_model
        # else:
        #     raise ValueError("Pretrained Model is not subclass of torch.nn.Module")

        # Model label
        self.model_name = config.get('model', None)
        if self.model_name is None:
            self.model_name = self.pretrained_model.__class__.__name__

        # We support two config modes
        # (1) 'autoq' dict in NNCF compression.initializer
        # (2) 'auto_quantization' as standalone dict in NNCF config
        if 'autoq' == self.config.nncf_config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {}):
            self.autoq_cfg = self.config.nncf_config.get('compression', {}).get('initializer', {}).get('precision')
        else:
            self.autoq_cfg = self.config.nncf_config.get('auto_quantization')
        self.finetune = self.autoq_cfg['finetune']

        self.tie_quantizers = True # Default to associate same precision of quantizers that feed into the GEMM compute
        if 'tie_quantizers' in self.autoq_cfg:
            self.tie_quantizers = self.autoq_cfg['tie_quantizers']

        self.skip_wall = False # Default to enable resource constraints
        if 'skip_wall' in self.autoq_cfg:
            self.tie_quantizers = self.autoq_cfg['skip_wall']

        # Action space boundary - need to revise to work with discrete
        self.action_bound = self._get_action_space(self.autoq_cfg)

        # TODO (Design): How to generalize evaluation and train in general? tasks could have different metric and loss function
        self._evaluate_pretrained_model()

        # Quantizer Master Table Creation
        qtracer = QuantizerTracer()
        # qtable index is QID in string prepended with its quantize node id
        qtable = qtracer.get_qtable(self.qctrl, self.qmodel)
        qgroups = qtracer.get_quantizer_groups(qtable)
        self.qgroups = qgroups
        if len(qtable) != len(self.qctrl.all_quantizations):
            logger.warning("[Warning][Q.Env] qtable has {} quantizers while qctrl has {} quantizers".format(
                len(qtable), len(self.qctrl.all_quantizations)
            ))

            diff = set(list(map(str, self.qctrl.all_quantizations.keys()))) ^ set(qtable.qid.values)
            for i, qidstr in enumerate(diff):
                logger.warning("[Warning] Extra quantizer {}: {}".format(i, qidstr))
    
        # Create master dataframe to keep track of quantizable layers and thier attributes, a.k.a state embedding
        self.master_df, self.state_list = self._get_state_space(self.qctrl, self.qmodel, qtable)
        
        # Workaround to set the min and max of action before fitting the minmaxscaler
        self.master_df['prev_action'][0]=self.max_bit
        self.master_df['prev_action'][-1]=self.min_bit
        
        # MinMaxScaler for State Embedding
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.master_df[self.state_list])

        # Log Master Table to run folder        
        self.master_df.drop('state_module', axis=1).to_csv(osp.join(self.config.log_dir, self.model_name + "_quantizable_state_table.csv"), index_label="nodestr")

        # Log qgroups
        with open(osp.join(self.config.log_dir, self.model_name + "_quantizer_groups.json"), "w") as qgroups_log:
            json.dump(self.qgroups, qgroups_log, indent=4)

        # Model Size Calculation
        self.orig_model_size   = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.float_bit  #in bit unit
        self.min_model_size    = sum(self.master_df['param']*self.master_df.is_wt_quantizer)*self.min_bit    # This variable has only been used once, just to ensure size constrainer doesnt go below than this
        self.target_model_size = self.orig_model_size*self.compress_ratio

        # init reward
        self.best_reward = -math.inf #TODO: move reward to search manager
        self.reset()
        # End of QuantizationEnv.__init__()
        # ----------------------------------------------------------------------------------------

    def _evaluate_pretrained_model(self):
        # TODO: How do we generalize evaluation metrics?
        # Currently Expect only a scalar output

        logger.info("[Q.Env] Evaluating Pretrained Model")
        self.qctrl.disable_weight_quantization()
        self.qctrl.disable_activation_quantization()

        with torch.no_grad():
            self.orig_acc = self.validate_fn(self.val_loader, self.qmodel, self.criterion, self.config)
            logger.info("Pretrained accuracy: {:.2f}".format(self.orig_acc))
        
        self.qctrl.enable_weight_quantization()
        self.qctrl.enable_activation_quantization()
        self.qmodel.rebuild_graph()

    def reset(self):
        self.collected_strategy=[]
        self.strategy=[]
        self.master_df['action']=0
        self.master_df['prev_action']=0
        self.master_df['unconstrained_action']=0

    def _calc_quantized_model_size(self): # in bit
        assert len(self.strategy) == len(self.master_df) # This function is only allowed when all actions are predicted        
        return sum(self.master_df['param'] * self.master_df.is_wt_quantizer * self.master_df['action'])
    
    def _expand_collected_strategy(self, collected_strategy):
        def find_group_members(qgroups, nodestr):
            for qgroup in qgroups:
                if nodestr in qgroup:
                    return qgroup
            return []

        grouped_strategy_map = OrderedDict(zip(list(self.master_df.index[self.master_df.is_pred]), collected_strategy))
        for nodestr, action in grouped_strategy_map.items():
            self.master_df.loc[nodestr, 'action']=action # This is needed for dangling quantizer
            for _nodestr in find_group_members(self.qgroups, nodestr):
                self.master_df.loc[_nodestr, 'action']=action

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

                            for qgroup in self.qgroups:
                                if nodestr in qgroup:
                                    for each_nodestr in qgroup:
                                        self.master_df.loc[each_nodestr, "action"] = new_bit
                                    break

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
        return (acc - self.orig_acc) * 0.1 

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
        step_cfg = self.config.nncf_config['compression']

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
        self.config.nncf_config['compression']=step_cfg
        return True 
        
    # TODO: # Hardcoded to Int2/4/8 for now, should open up the space later to FP32, FP16
    def _get_action_space(self, cfg):
        precision_set = cfg.get('precision_set', None)

        if precision_set is not None:
            pass
            # TODO: define the precision convention and parser here
        else:
            pass
        
        self.compress_ratio = cfg['compress_ratio'] if 'compress_ratio' in cfg else 0.15
        self.float_bit = 32.0
        self.min_bit = 2
        self.max_bit = 8
                
        return (self.min_bit, self.max_bit)


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
       
    # def _create_quantization_pipeline(self, nncf_config):
    #     _pretrained_model = deepcopy(self.pretrained_model)
    #     compression_ctrl, compressed_model = create_compressed_model(_pretrained_model, nncf_config)
    #     return compression_ctrl, compressed_model


    def _run_quantization_pipeline(self, finetune):
        self._adaptbn(self.config)
        
        if finetune:
            # TODO: need to fix finetune loop
            top1, top5 = self._finetune(self.qctrl, self.qmodel)
            logger.info("[Q.Env] Post-Tuned: Top1@ {:.3f}, Top5@ {:.3f}".format(top1, top5))
        else:
            with torch.no_grad():
                quantized_acc = self.validate_fn(self.val_loader, self.qmodel, self.criterion, self.config)
                logger.info("[Q.Env] Post-Init: {:.3f}".format(quantized_acc))

        return quantized_acc


    def _adaptbn(self, config):
        train_mode = self.qmodel.training
        if not train_mode:
            self.qmodel.train()

        self.qctrl.run_batchnorm_adaptation(self.qctrl.quantization_config)
        
        if not train_mode:
            self.qmodel.eval()


    def _finetune(self, compression_ctrl, compressed_model):
        best_compression_level = CompressionLevel.NONE

        params_to_optimize = get_parameter_groups(compressed_model, self.config)
        optimizer, lr_scheduler = make_optimizer(params_to_optimize, self.config)

        for epoch in range(self.config.start_epoch, self.config.epochs):
            self.config.cur_epoch = epoch

            # train for one epoch
            self.train_epoch_fn(
                self.train_loader,
                compressed_model, 
                self.criterion, 
                optimizer, 
                compression_ctrl,
                epoch,
                self.config,
                False)

            # Learning rate scheduling should be applied after optimizer’s update
            best_acc1 = 0 # what is the usage here?
            lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_acc1)

            # update compression scheduler state at the end of the epoch
            compression_ctrl.scheduler.epoch_step()

            # compute compression algo statistics
            stats = compression_ctrl.statistics()

            acc1 = best_acc1
            if epoch % self.config.test_every_n_epochs == 0:
                # evaluate on validation set
                top1, top5 = self.validate_fn(self.val_loader, compressed_model, self.criterion, self.config)
        return top1, top5














