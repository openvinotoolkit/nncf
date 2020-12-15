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

class ModelSizeCalculator:
    FLOAT_BITWIDTH = 32
    def __init__(self, qmodel, per_quantizer_bw_space):
        self._bw_space_map = OrderedDict()
        self._nparam_map = OrderedDict()
        for qid, bw_space in per_quantizer_bw_space.items():
            if isinstance(qid, WeightQuantizerId):
                self._bw_space_map[qid] = bw_space
                m = qmodel.get_module_by_scope(qid.scope)
                self._nparam_map[qid] = np.prod(m.weight.size())    

        self.min_model_size, self.max_model_size = \
            self._calc_min_max_model_size()

        self.fp_model_size = self.get_uniform_bit_model_size(ModelSizeCalculator.FLOAT_BITWIDTH)

    def __call__(self, per_quantizer_bw):
        return self.get_model_size(per_quantizer_bw)

    def _calc_min_max_model_size(self):
        nparam_array = np.array(list(self._nparam_map.values()))
        min_size = np.sum( nparam_array * np.array(list(map(min, self._bw_space_map.values()))) )
        max_size = np.sum( nparam_array * np.array(list(map(max, self._bw_space_map.values()))) )
        return min_size, max_size

    def get_uniform_bit_model_size(self, uniform_bitwidth: int):
        return np.sum( np.array(list(self._nparam_map.values())*uniform_bitwidth) )

    def get_model_size(self, per_quantizer_bw: OrderedDict):
        model_size = 0
        for qid, nparam in self._nparam_map.items():
            if qid in per_quantizer_bw:
                model_size += nparam * per_quantizer_bw[qid]
            else:
                logger.warn("[ModelSizeCalculator] Missing Bitwidth of QID: {}, using {} bits"
                                .format(str(qid), ModelSizeCalculator.FLOAT_BITWIDTH))
                model_size += nparam * ModelSizeCalculator.FLOAT_BITWIDTH
        return model_size

    def get_model_size_ratio(self, per_quantizer_bw: OrderedDict):
        return self.get_model_size(per_quantizer_bw)/self.fp_model_size


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
        # TODO: Clean up
        if 'autoq' == self.config.get('compression', {}).get('initializer', {}).get('precision', {}).get('type', {}):
            self.autoq_cfg = self.config.get('compression', {}).get('initializer', {}).get('precision')
        else:
            raise ValueError("Missing/Invalid Config of Precision Initializer. "
                             "Pls review config['compression']['initializer']['precision']")

        # Set target compression ratio
        self.compression_ratio = self.autoq_cfg.get('compression_ratio', 0.15)
        
        # Bool to disable hard resource constraint
        self.skip_constraint = False
        if 'skip_constraint' in self.autoq_cfg:
            self.skip_constraint = self.autoq_cfg['skip_constraint']

        # Bool to enable fine-tuning in each episode. Placeholder for now
        self.finetune = False
        if 'finetune' in self.autoq_cfg:
            self.finetune = self.autoq_cfg['finetune']

        # Configure search space for precision according to target device
        if self.hw_cfg_type is None:
            self.model_bitwidth_space = self.autoq_cfg.get('bits', [2, 4, 8])
        elif self.hw_cfg_type is HWConfigType.VPU:
            self.model_bitwidth_space = self.qctrl._hw_precision_constraints.get_all_unique_bits()
        self.model_bitwidth_space = sorted(list(self.model_bitwidth_space))
  
        # Create mapping of QuantizerId to its bitwidth space (per quantizer bitwidth space)
        self.bw_space_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())
        if self.hw_cfg_type is None:
            for qid in self.bw_space_map.keys():
                self.bw_space_map[qid] = self.model_bitwidth_space
        else:
            assert hasattr(self.qctrl._hw_precision_constraints, '_constraints'), \
                "feasible bitwidth per quantizer not found"
            for qid, bw_space in self.qctrl._hw_precision_constraints._constraints.items():
                self.bw_space_map[qid] = sorted(list(bw_space))

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
        self.model_size_calculator = ModelSizeCalculator(self.qmodel, self.bw_space_map)
        self.orig_model_size       = self.model_size_calculator.fp_model_size
        self.min_model_size        = self.model_size_calculator.min_model_size
        self.max_model_size        = self.model_size_calculator.max_model_size
        self.target_model_size     = self.orig_model_size*self.compression_ratio

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
        self.master_df['action']=max(self.model_bitwidth_space)
        self.master_df['prev_action']=0
        self.master_df['unconstrained_action']=0


    def _create_quantizer_table(self):
        # Create a mapping of qid to its adjacent quantizer group id
        adjq_gid_map = OrderedDict.fromkeys(self.qctrl.all_quantizations.keys())
        for qid, qmod in self.qctrl.all_quantizations.items():
            adjq_gid_map[qid] = self._groups_of_adjacent_quantizers.get_group_id_for_quantizer(qmod)

        assert len(set(self.bw_space_map.keys()) - set(adjq_gid_map.keys())) == 0, \
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
            d[idx_str]['bw_space']     = self.bw_space_map[qid]
            
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
        
        # Annotate a min and a max value in prev_action before minmaxscaler fitting
        master_df['prev_action'][ 0] = max(self.model_bitwidth_space)
        master_df['prev_action'][-1] = min(self.model_bitwidth_space)

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


    def _get_quantizer_bitwidth(self):
        assert len(set(self.model_bitwidth_space) - set(self.master_df.action.values)) >= 0, \
            "there is bitwidth choice not within model bitwidth space"
        return OrderedDict(zip(self.master_df.qid_obj, self.master_df.action))


    def _constrain_model_size(self, skip=False):
        # This function acts on self.strategy and return self.strategy
        def lower_bitwidth(bw, bw_space):
            return bw_space[bw_space.index(bw)-1] if bw_space.index(bw) > 0 else bw

        if skip is not True:
            self.master_df['unconstrained_action']=self.master_df['action']

            current_model_size = self.model_size_calculator(self._get_quantizer_bitwidth())

            while self.min_model_size < current_model_size and self.target_model_size < current_model_size:
                for i, nodestr in enumerate( reversed(self.master_df.index.tolist()) ):
                    if self.master_df.loc[nodestr, "is_wt_quantizer"]: 
                        bw_choice, bw_space = self.master_df.loc[nodestr, ['action', 'bw_space']]
                        new_bw = lower_bitwidth(bw_choice, bw_space)
                        self.master_df.loc[nodestr, "action"] = new_bw if new_bw != bw_choice else bw_choice

                    current_model_size = self.model_size_calculator(self._get_quantizer_bitwidth())
                    if current_model_size <= self.target_model_size:
                        break
        else:
            logger.info("[Q.Env] Skipping Model Size Constraint")

        self.strategy = self.master_df['action']
        return self.strategy

    def reward(self, acc, model_ratio):
        return (acc - self.pretrained_score) * 0.1 

    def step(self, action):
        def is_final_step():
            return len(self.collected_strategy) == len(self.master_df)
        
        # Ensure action is in the quantizer's bitwidth space
        current_bw_space = self.master_df.bw_space[len(self.collected_strategy)]
        if action not in current_bw_space:
            closest_bw_idx = np.argmin(np.abs(action - np.array(current_bw_space)))
            action = current_bw_space[closest_bw_idx]
        
        self.collected_strategy.append(action)

        if not is_final_step():
            info_set = {}
            reward = 0
            self.set_next_step_prev_action(len(self.collected_strategy), action)
            obs = self.get_normalized_obs( len(self.collected_strategy) )
            done = False
            return obs, reward, done, info_set
        else:
            return self.evaluate_strategy(self.collected_strategy, skip_constraint=self.skip_constraint)
        
    def evaluate_strategy(self, collected_strategy, skip_constraint=True):
        self.master_df['action'] = collected_strategy
        self.strategy = self.master_df['action']
        
        if skip_constraint is not True:
            self.strategy = self._constrain_model_size()
        
        assert len(self.strategy) == len(self.master_df)

        # Quantization
        self.apply_actions(self.strategy)
        
        current_model_size  = self.model_size_calculator(self._get_quantizer_bitwidth())
        current_model_ratio = self.model_size_calculator.get_model_size_ratio(self._get_quantizer_bitwidth())

        for idx, qid in zip(self.master_df.index, self.master_df['qid']):
            logger.info("[Q.Env] {:50} | {}".format(
                str(self.qctrl.all_quantizations[ find_qid_by_str(self.qctrl, qid) ]),
                idx))

        quantized_score = self._run_quantization_pipeline(finetune=self.finetune)
        reward = self.reward(quantized_score, current_model_ratio)
        
        info_set = {'model_ratio': current_model_ratio, 'accuracy': quantized_score, 'model_size': current_model_size}

        if reward > self.best_reward:
            self.best_reward = reward
            prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, model_ratio: {:.3f}, model_size(mb): {:.3f}'.format(
                self.strategy, self.best_reward, quantized_score, current_model_ratio, current_model_size/8000000))

        obs = self.get_normalized_obs( len(collected_strategy)-1 )
                                                    
        done = True
        return obs, reward, done, info_set

    def set_next_step_prev_action(self, idx, action):
        self.master_df.loc[self.master_df.index[idx], 'prev_action'] = action


    def get_normalized_obs(self, idx):
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
                # if we use scope for non-weight quantizer, we would risk masking out the quantizers within the scope
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







