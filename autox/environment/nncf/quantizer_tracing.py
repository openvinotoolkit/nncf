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
from functools import partial
from shutil import copyfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10, CIFAR100

from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import configure_distributed
from examples.common.example_logger import logger
from examples.common.execution import ExecutionMode, get_device, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.sample_config import SampleConfig, create_sample_config
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, make_additional_checkpoints, get_name, is_staged_quantization, print_statistics, \
    is_pretrained_model_requested
from examples.common.utils import write_metrics

# from nncf import create_compressed_model
# We use a local version here to bypass the initialization step
from nncf import create_compressed_model

from nncf.compression_method_api import CompressionLevel
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.initialization import register_default_init_args
from nncf.utils import manual_seed, safe_thread_call, is_main_process
from datetime import datetime

from nncf import NNCFConfig

from collections import OrderedDict, Counter
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId, InputQuantizerId, FunctionQuantizerId
from nncf.dynamic_graph.context import Scope
from natsort import natsorted
import pandas as pd
from pandas.core.common import flatten
import numpy as np
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId, InputQuantizerId, FunctionQuantizerId

# Helper Function ==================================

class QuantizerTracer():
    def __init__(self):
        # TODO: should we keep a reference to quantization_controller and quantized Model?
        pass

    def get_qtable(self, quantization_controller, quantized_model):
        g=quantized_model.get_graph()

        # Tag input quantizers to GEMM
        gemm_quantizer_dict = get_gemm_with_input_quantizers(quantization_controller, quantized_model)
        # gemm_quantizer_dict
        # Key => GEMM Module in NNCF Scope
        # val['WQ']  => list of tuple (weight quantize nx_nodekey, WeightQuantizerId)
        # val['NWQ'] => list of tuple (non-weight quantize nx_nodekey, WeightQuantizerId)

        # Get dangling quantizers
        untagged_QId_list = get_untagged_quantizer(quantization_controller, gemm_quantizer_dict)

        # = Group dependent GEMM by common non-weight quantizer ====
        # store tagged non-weight quantizers into a tuple list
        gemm_tagged_nonwt_quantizer_tuple_list = []
        for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
            gemm_tagged_nonwt_quantizer_tuple_list.extend(gemm_dict["NWQ"])

        # extract tagged non-weight quantizers into a list
        gemm_tagged_nonwt_qid_list = list(map(lambda x: x[1], gemm_tagged_nonwt_quantizer_tuple_list))

        # count occurence of non-weight quantizer
        meter=Counter(gemm_tagged_nonwt_qid_list)

        # quantizers which have occurred once mean they fan into multiple gemm
        # create a dict with these quantizers as key and list of dependent gemm (scope objects)
        grouped_gemm_dict=OrderedDict()
        for nonwt_qid, count in meter.items():
            if count > 1:
                grouped_gemm_list=[]
                for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
                    for qnodekey, qid in gemm_dict['NWQ']:
                        if nonwt_qid == qid:
                            grouped_gemm_list.append(str(gemm_scope))
                grouped_gemm_dict[nonwt_qid]=grouped_gemm_list
        # = End of Grouping dependent GEMM ===========================

        # = Q. Table creation ========================================
        # Create dictionary structure for pandas dataframe
        d = OrderedDict()
        for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
            wt_quantize_nodekey, wqid = gemm_dict['WQ']
            non_wt_q_tuple_list = gemm_dict['NWQ']
            gemm_nx_nodekey = gemm_dict['nx_nodekey']

            # populate weight quantizer
            q_nx_nodeid = wt_quantize_nodekey.split()[0]
            idx_str = '-'.join([q_nx_nodeid, wqid])
            if idx_str in d:
                raise ValueError("There should not be existence of GEMM weight quantizer in d because of uniquification earlier")
            else:
                d[idx_str] = OrderedDict()
                d[idx_str]['qid'] = wqid
                d[idx_str]['q_nx_nodeid'] = q_nx_nodeid
                d[idx_str]['q_nx_nodekey'] = wt_quantize_nodekey
                d[idx_str]['gemm_nx_nodekey'] = [gemm_nx_nodekey]
                wqid_obj = find_qidobj(quantization_controller, wqid)
                d[idx_str]['state_scope'] = wqid_obj.scope

            # populate non-weight quantizer
            for tup in non_wt_q_tuple_list:
                non_wt_quantize_nodekey, nwqid = tup

                q_nx_nodeid = non_wt_quantize_nodekey.split()[0]
                idx_str = '-'.join([q_nx_nodeid, nwqid])
                
                if idx_str in d:
                    d[idx_str]['gemm_nx_nodekey'].append(gemm_nx_nodekey)
                else:
                    d[idx_str] = OrderedDict()
                    d[idx_str]['qid'] = nwqid
                    d[idx_str]['q_nx_nodeid'] = q_nx_nodeid
                    d[idx_str]['q_nx_nodekey'] = non_wt_quantize_nodekey
                    d[idx_str]['gemm_nx_nodekey'] = [gemm_nx_nodekey]
                    nwqid_obj = find_qidobj(quantization_controller, nwqid)
                    d[idx_str]['state_scope'] = nwqid_obj.ia_op_exec_context.scope_in_model

        # populate dangling quantizer
        # for qid in untagged_QId_list:
        untagged_qidobj_list = list(map(lambda x: find_qidobj(quantization_controller, x), untagged_QId_list))
        assert len(untagged_qidobj_list) == len(untagged_QId_list), "Assumption Breaks on untagged qid"
        
        for i, qid in enumerate(untagged_qidobj_list):
            if str(qid) in qid_skiplist:
                continue
            nx_node = g.get_nx_node_by_key(g.get_node_id_by_iap_context(qid.ia_op_exec_context))
            child_nodes = list(g.get_successors(nx_node['key']))
            assert len(child_nodes) == 1, "multi fan-out, would this be an issue?"

            untagged_quantize_nodekey = child_nodes[0]

            q_nx_nodeid = untagged_quantize_nodekey.split()[0]
            idx_str = '-'.join([q_nx_nodeid, str(qid)])
                
            if idx_str in d:
                raise ValueError("How could a dangling quantizer already exist?")
            else:
                d[idx_str] = OrderedDict()
                d[idx_str]['qid'] = str(qid)
                d[idx_str]['q_nx_nodeid'] = q_nx_nodeid
                d[idx_str]['q_nx_nodekey'] = untagged_quantize_nodekey
                d[idx_str]['gemm_nx_nodekey'] = []
                d[idx_str]['state_scope'] = qid.ia_op_exec_context.scope_in_model
        df = pd.DataFrame.from_dict(d,orient='index')
        df = df.loc[natsorted(df.index)]

        return df

    def get_quantizer_groups(self, qtable):
        
        quantizer_groups=[]
        # nested gemm 
        for nodestr, gemm_group in qtable.gemm_nx_nodekey[qtable.gemm_nx_nodekey.apply(lambda x: len(x) > 1)].items():
            group_member_indices = []
            for gemm in gemm_group:
                for _nodestr, _gemm_nx_nodekey in qtable.gemm_nx_nodekey.items():
                    if gemm in _gemm_nx_nodekey:
                        group_member_indices.append(_nodestr)
            group_member_indices = list(dict.fromkeys(group_member_indices)) # remove duplicates
            quantizer_groups.append(group_member_indices)

        # quantizers to gemm
        remaining_quantizer_indices = list(set(qtable.index.to_list()) - set(natsorted(list(flatten(quantizer_groups)))))
        remainder_df = qtable.loc[pd.Index(remaining_quantizer_indices)]
        
        for nodestr, gemm_group in remainder_df.gemm_nx_nodekey[
                                        remainder_df.qid.apply(lambda x: 'module_weight' in x) & 
                                        remainder_df.gemm_nx_nodekey.apply(lambda x: len(x) == 1)
                                        ].items():
            gemm=gemm_group[0]
            group_member_indices = []
            for _nodestr, _gemm_nx_nodekey in remainder_df.gemm_nx_nodekey.items():
                if gemm in _gemm_nx_nodekey:
                        group_member_indices.append(_nodestr)
            group_member_indices = list(dict.fromkeys(group_member_indices)) # remove duplicates
            quantizer_groups.append(group_member_indices)

        qgroup_dict = OrderedDict()
        for g in quantizer_groups:
            sorted_g = natsorted(g)
            qgroup_dict[sorted_g[0]]=sorted_g
        qgroups = [qgroup_dict[k] for k in natsorted(qgroup_dict.keys())]

        return qgroups


def find_qidobj(quantization_controller, qid_str):
    for _qid, _q in quantization_controller.all_quantizations.items():
        if qid_str == str(_qid):
            return _qid

def uptrace_quantize_nncfnode(nncfgraph, nncfnode, uptrace_node_list):
    g=nncfgraph
    _nodekey = g.get_node_key_by_id(nncfnode.node_id)
    if "Quantizer" in _nodekey:
        # add quantize node to the list
        uptrace_node_list.append(nncfnode)
    else:
        # uptrace for non-quantize node
        input_nncfnodes = g.get_previous_nodes(nncfnode)
        for input_nncfnode in input_nncfnodes:
            uptrace_quantize_nncfnode(g, input_nncfnode, uptrace_node_list)
    return uptrace_node_list

def get_non_weight_quantizer_id(nncfgraph, nncfnode):
    g=nncfgraph
    nodekey=g.get_node_key_by_id(nncfnode.node_id)
    if 'quantize' not in nodekey:
        raise ValueError("This is not a quantize node")
    
    # Assumption: non-weight quantizer id usually follows
    # the scope of predecessor node (from observation thus far)
    innodes = g.get_previous_nodes(nncfnode)

    if len(innodes) != 1:
        raise ValueError("Special Case, Assumption breaks, more than a single parent nodes")

    innodekey=g.get_node_key_by_id(innodes[0].node_id)
    in_nxnode=g.get_nx_node_by_key(innodekey)

    return NonWeightQuantizerId(in_nxnode['op_exec_context'].input_agnostic)

def NWQId_exist_Qctrl(NWQId, quantization_controller):
    for key in quantization_controller.non_weight_quantizers.keys():
        if NWQId.__str__() == key.__str__():
            return True
    return False
# End of Helper Functions ==================================


# Assumption: All GEMM has only two parent nodes (1) parameters loading node (2) Any Operation node in DAG

# Extra Info:
# NNCFNode is a base class that capture node_id and IA_OP_CTX. 
# NNCFGraph is networkx digraph, each node is nx_node. To convert nx_node to NNCFNode, use NNCFGraph._nx_node_to_nncf_node(nx_node)
# we refer node as nncfnode and nx_node as networkx node

# gemm_module as key
# WQ => (nodekey, quantizer)
# NWQ => [(nodekey, quantizer),(nodekey, quantizer), ...]
# skiplist store the nncf_module that only exist for training purpose; tracer is not able to prune the nncf_modules to contain eval-only module
gemm_skiplist =[
    'ICNet/CascadeFeatureFusion[cff42]/NNCFConv2d[classifier]',
    'ICNet/CascadeFeatureFusion[cff421]/NNCFConv2d[classifier]'
]
qid_skiplist= [
    'ICNet/CascadeFeatureFusion[cff42]/NNCFConv2d[classifier]module_weight',
    'ICNet/CascadeFeatureFusion[cff421]/NNCFConv2d[classifier]module_weight',
]
def get_gemm_with_input_quantizers(quantization_controller, quantized_network):
    g=quantized_network.get_graph() # compressed nncf graph instance

    gemm_scope_list= quantized_network.get_nncf_module_scopes()

    gemm_quantizer_dict=OrderedDict()

    print("= Trace input quantizer to GEMM =======")

    for gemm in gemm_scope_list:
        
        if str(gemm) in gemm_skiplist:
            continue

        print("GEMM Scope | {}".format(str(gemm)))

        gemm_quantizer_dict[gemm]=OrderedDict()
        qnode_list=[]
        
        for nx_node in g.get_op_nodes_in_scope(gemm):
            # Node with 'UpdateWeight' keyword is the weight quantize node in NNCFGraph 
            if 'UpdateWeight' in nx_node['key']:
                
                WQId = WeightQuantizerId(gemm)
                for key in quantization_controller.weight_quantizers.keys():
                    if str(WQId) == str(key): # string comparison is required as the object reference differs
                        gemm_quantizer_dict[gemm]['WQ']=(nx_node['key'], str(WQId))
                        print("==> Found W Quantizer | {}".format(str(WQId)))

            # Node without Quantizer keyword is a non-quantize node fanning into the GEMM,
            # We need to trace up to quantize nodes 
            if 'Quantizer' not in nx_node['key']:
                print("\t# Need to trace up for node => {}".format(nx_node['key']))
                
                # Input nodes to current nx_node, note that the returned node is in NNCFNode
                input_node_list = g.get_previous_nodes(g._nx_node_to_nncf_node(nx_node))

                # Embedding only take in a single input, the index key
                if len(input_node_list) !=2:
                    if len(input_node_list)==1 and 'NNCFEmbedding' in nx_node['key']:
                        pass
                    else:
                        raise ValueError("Number of Input nodes to GEMM should be 2, assumption breaks, pls debug")
                    
                # Filter weight quantizer node as it has been handled earlier
                uptrace_node_list=[]
                for nncf_node in input_node_list:
                    nx_nodekey = g.get_node_key_by_id(nncf_node.node_id)
                    if 'UpdateWeight' in nx_nodekey:
                        print("\t# Skipping weight quantize node | {}".format(nx_nodekey))
                    else:
                        print("\t# Adding node to uptrace list | {}".format(nx_nodekey))
                        uptrace_node_list.append(nncf_node)
                
                # Recursively trace until a non-weight quantizer ====================
                gemm_quantizer_dict[gemm]['NWQ']=[]
                
                print("\t# Uptracing the following nodes: {}".format(list(map(lambda x: x.node_id, uptrace_node_list))))

                for i, innode in enumerate(uptrace_node_list):
                    print("\t\t# Tracing node_id | {}".format(innode.node_id))
                    
                    quantize_innode_list=[]
                    quantize_innode_list=uptrace_quantize_nncfnode(g, innode, quantize_innode_list)
            
                    # following could be disable, only for inspection
                    for innode in quantize_innode_list:
                        innodekey = g.get_node_key_by_id(innode.node_id)
                        print("\t\t\t# quantize node | {}".format(innodekey))

                    # Following is to extract the Non-Weight Quantizer Id
                    # Assumption: In order to get the corresponding Quantizer Id of the node
                    # we need to trace up a level because the quantizer Id is created based on that scope
                    # For Input Quantizer Id: it uses the subsequent module as argument for InputQuantizerId creator

                    for innode in quantize_innode_list:
                        innodekey = g.get_node_key_by_id(innode.node_id)

                        # Special handling for input
                        if 'UpdateInputs' in innodekey:
                            #  invalidated if input to multiple gemm
                            if len(list(g.get_successors(innodekey))) != 1:
                                raise ValueError("Model Input fans in to multiple nodes, assumption breaks, pls debug")
                            
                            input_module_nodekey = list(g.get_successors(innodekey))[0]
                            IQId = InputQuantizerId(g.get_nx_node_by_key(input_module_nodekey)['op_exec_context'].input_agnostic)
                            
                            if NWQId_exist_Qctrl(IQId, quantization_controller):
                                gemm_quantizer_dict[gemm]['NWQ'].append((innodekey, str(IQId)))
                                print("==> Found NW Quantizer | {}".format(str(IQId)))
                            else:
                                raise ValueError("IQId not found, Pls Debug")
                        else:
                            NWQId=get_non_weight_quantizer_id(g, innode)
                            
                            if NWQId_exist_Qctrl(NWQId, quantization_controller):
                                gemm_quantizer_dict[gemm]['NWQ'].append((innodekey, str(NWQId)))
                                print("==> Found NW Quantizer | {}".format(str(NWQId)))
                            else:
                                raise ValueError("NWQId not found, Pls Debug")
                # [End OF] Recursively trace until a non-weight quantizer ====================
    
    # Bookkeep GEMM Quantize Nodekey
    for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
        wt_quantize_nodekey, wqid = gemm_dict['WQ']
        
        child_nx_nodes = list(g.get_successors(wt_quantize_nodekey))
        if len(child_nx_nodes) != 1:
            raise ValueError("Child of weight quantize node should be GEMM node, assumption Breaks, pls debug")
        
        gemm_nx_nodekey = child_nx_nodes[0]        
        gemm_quantizer_dict[gemm_scope]['nx_nodekey']=gemm_nx_nodekey

    print("\n")
    return gemm_quantizer_dict

def get_untagged_quantizer(quantization_controller, gemm_quantizer_dict):
    # Find untagged quantizer
    QId_list=list(map(str,quantization_controller.all_quantizations.keys()))

    gemm_tagged_quantizer_tuple_list = [] 
    for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
        gemm_tagged_quantizer_tuple_list.append(gemm_dict["WQ"])
        gemm_tagged_quantizer_tuple_list.extend(gemm_dict["NWQ"])

    gemm_tagged_quantizer_nx_nodekey_list=list(map(lambda x:x[0], gemm_tagged_quantizer_tuple_list))
    gemm_tagged_QId_list=list(map(lambda x:x[1], gemm_tagged_quantizer_tuple_list))

    untagged_QId_list = list(set(QId_list)-set(gemm_tagged_QId_list))
    return untagged_QId_list

# MAIN ====================================================================================
if __name__ == '__main__':
        
    # cfgfile="./cfg/vgg11_imagenet_autoq.json"
    # cfgfile="./cfg/resnet18_imagenet_autoq.json"
    # cfgfile="./cfg/mobilenet_v2_imagenet_autoq.json"
    cfgfile = "./cfg/unet_camvid_autoq.json"

    nncf_config = NNCFConfig.from_json(cfgfile)
    arch=nncf_config['model']

    model = load_model(arch,
                        pretrained=False, 
                        num_classes=nncf_config.get('num_classes', 0), # for error out when num_classes not defined in config
                        model_params=nncf_config.get('model_params', {}))

    # model = models.__dict__[arch](pretrained=True)
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    print("= List all quantizers <Quantizer> | <QuantizerId> =======")
    for QId, Q in compression_ctrl.all_quantizations.items():
        print("{} | {}".format(Q, QId))
    print("= [End of] List all quantizers =======","\n")

    gemm_quantizer_dict = get_gemm_with_input_quantizers(compression_ctrl, compressed_model)

    # gemm_quantizer_dict
    # Key => GEMM Module in NNCF Scope
    # val['WQ']  => list of tuple (weight quantize nx_nodekey, WeightQuantizerId)
    # val['NWQ'] => list of tuple (non-weight quantize nx_nodekey, WeightQuantizerId)

    for gemm_scope, val in gemm_quantizer_dict.items():
        print('GEMM: {}'.format(gemm_scope))
        print("WQ  : {}".format(val['WQ']))
        print("NWQ : {}".format(val['NWQ']))
        
        print("\n")

    untagged_QId_list = get_untagged_quantizer(compression_ctrl, gemm_quantizer_dict)

    # This section is only for visual inspection
    g=compressed_model.get_graph()

    gemm_tagged_quantizer_tuple_list = [] 
    for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
        gemm_tagged_quantizer_tuple_list.append(gemm_dict["WQ"])
        gemm_tagged_quantizer_tuple_list.extend(gemm_dict["NWQ"])

    gemm_tagged_quantizer_nx_nodekey_list=list(map(lambda x:x[0], gemm_tagged_quantizer_tuple_list))
    gemm_tagged_QId_list=list(map(lambda x:x[1], gemm_tagged_quantizer_tuple_list))

    dangling_nodekeys = []

    for untagged_QId in untagged_QId_list:
        print("Untagged QId: {}".format(untagged_QId))
        scope='/'.join(untagged_QId.split("/")[:-1])
        nodes_in_scope = g.get_op_nodes_in_scope(Scope.from_str(scope))
        
        quantize_nodekeys=[]
        for node in nodes_in_scope:
            if 'quantize' in node['key']:
                quantize_nodekeys.append(node['key'])
                
        if len(quantize_nodekeys) == 1:
            print("\t", quantize_nodekeys[0])
            
            dangling_nodekeys.append(quantize_nodekeys[0])
        else:
            print("Warning") # meaning there are multiple quantize node in this scope
            for nxkey in quantize_nodekeys:
                if nxkey not in gemm_tagged_quantizer_nx_nodekey_list:
                    print(nxkey)
                    dangling_nodekeys.append(nxkey)
        print("\n")

    for i, nodekey in enumerate(natsorted(dangling_nodekeys)):
        print("UnTagged Quantize Node {} | {}".format(i, nodekey))

    #### Group dependent GEMM by common non-weight quantizer

    gemm_tagged_nonwt_quantizer_tuple_list = []
    for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
        gemm_tagged_nonwt_quantizer_tuple_list.extend(gemm_dict["NWQ"])

    gemm_tagged_nonwt_qid_list = list(map(lambda x: x[1], gemm_tagged_nonwt_quantizer_tuple_list))

    meter=Counter(gemm_tagged_nonwt_qid_list)

    grouped_gemm_dict=OrderedDict()

    for nonwt_qid, count in meter.items():
        if count > 1:
            grouped_gemm_list=[]
            print(nonwt_qid)
            for gemm_scope, gemm_dict in gemm_quantizer_dict.items():
                for qnodekey, qid in gemm_dict['NWQ']:
                    if nonwt_qid == qid:
                        grouped_gemm_list.append(str(gemm_scope))
            grouped_gemm_dict[nonwt_qid]=grouped_gemm_list

    for qid, gemm_list in grouped_gemm_dict.items():
        for qnodekey, _qid in gemm_tagged_nonwt_quantizer_tuple_list:
            if _qid == qid:
                print("\nGrouped by: {}\n{}".format(qid,qnodekey))
                break

        for i, gemm in enumerate(gemm_list):
            print("{} | {}".format(i, gemm))