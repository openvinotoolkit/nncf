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

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from beta.nncf.api.compression import CompressionAlgorithmBuilder
from beta.nncf.api.compression import CompressionAlgorithmController
from beta.nncf.api.compression import CompressionScheduler
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.graph.converter import convert_layer_graph_to_nxmodel
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.graph.model_transformer import ModelTransformer
from beta.nncf.tensorflow.graph.transformations.commands import InsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import LayerWeight
from beta.nncf.tensorflow.graph.transformations.commands import LayerWeightOperation
from beta.nncf.tensorflow.graph.transformations.commands import RemovalCommand
from beta.nncf.tensorflow.graph.transformations.commands import TransformationPriority
from beta.nncf.tensorflow.graph.transformations.layout import TransformationLayout
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.graph.utils import get_custom_layers
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.graph.utils import get_weight_node_name
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.rb.loss import SparseLoss, SparseLossForPerLayerSparsity
from beta.nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from beta.nncf.tensorflow.sparsity.schedulers import SPARSITY_SCHEDULERS
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.utils.node import is_ignored


PRUNING_LAYERS = {
    'Conv1D': {'weight_attr_name': 'kernel'},
    'Conv2D': {'weight_attr_name': 'kernel'},
    'DepthwiseConv2D': {'weight_attr_name': 'depthwise_kernel'},
    'Conv3D': {'weight_attr_name': 'kernel'},
    'Conv2DTranspose': {'weight_attr_name': 'kernel'},
    'Conv3DTranspose': {'weight_attr_name': 'kernel'},
    'Dense': {'weight_attr_name': 'kernel'},
    'SeparableConv1D': {'weight_attr_name': 'pointwise_kernel'},
    'SeparableConv2D': {'weight_attr_name': 'pointwise_kernel'},
    'Embedding': {'weight_attr_name': 'embeddings'},
    'LocallyConnected1D': {'weight_attr_name': 'kernel'},
    'LocallyConnected2D': {'weight_attr_name': 'kernel'}
}


@TF_COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.ignored_scopes = self.config.get('ignored_scopes', [])

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        transformations = TransformationLayout()
        shared_nodes = set()

        for node_name, node in nxmodel.nodes.items():
            original_node_name, _ = get_original_name_and_instance_index(node_name)
            if node['type'] not in PRUNING_LAYERS \
                    or is_ignored(node_name, self.ignored_scopes) \
                    or original_node_name in shared_nodes:
                continue

            if node['is_shared']:
                shared_nodes.add(original_node_name)

            weight_attr_name = PRUNING_LAYERS[node['type']]['weight_attr_name']
            transformations.register(
                InsertionCommand(
                    target_point=LayerWeight(original_node_name, weight_attr_name),
                    callable_object=RBSparsifyingWeight(),
                    priority=TransformationPriority.SPARSIFICATION_PRIORITY
                ))

        return transformations

    def build_controller(self, model) -> CompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed
        """
        return RBSparsityController(model, self.config.get('params', {}))


class RBSparsityController(CompressionAlgorithmController):
    def __init__(self, target_model,
                 params):
        super().__init__(target_model)
        self._scheduler = None
        self._distributed = False
        self.sparsity_init = params.get('sparsity_init', 0)
        sparsity_level_mode = params.get("sparsity_level_setting_mode", "global")
        sparsifyed_layers = collect_wrapped_layers(target_model)
        self._check_sparsity_masks = params.get("check_sparsity_masks", False)
        if sparsity_level_mode == 'local':
            self._loss = SparseLossForPerLayerSparsity(sparsifyed_layers)
            self._scheduler = CompressionScheduler()
        else:
            self._loss = SparseLoss(sparsifyed_layers)  # type: SparseLoss
            schedule_type = params.get("schedule", "exponential")
            scheduler_cls = SPARSITY_SCHEDULERS.get(schedule_type)
            self._scheduler = scheduler_cls(self, params)

    def set_sparsity_level(self, sparsity_level, target_layer: NNCFWrapper = None):
        if target_layer is None:
            #pylint:disable=no-value-for-parameter
            self._loss.set_target_sparsity_loss(sparsity_level)
        else:
            self._loss.set_target_sparsity_loss(sparsity_level, target_layer)

    def freeze(self):
        self._loss.disable()
    '''
    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues
    '''
    def add_algo_specific_stats(self, stats):
        stats["target_sparsity_rate"] = self.loss.target_sparsity_rate
        #if self._distributed and self._check_sparsity_masks:
        #    stats["masks_consistents"] = self.check_distributed_masks()
        return stats
    # TODO: Ask about this piece of code
    #def set_sparsity_level_for_module(self, sparsity_level: float,
    #                                  target_sparsified_module_info: List[SparseModuleInfo]):
    #    # ???
    #    sparse_op = target_sparsified_module_info[0].operand
    #    self._loss.set_target_sparsity_loss_for_module(sparsity_level, sparse_op)

    def get_sparsity_init(self):
        return self.sparsity_init
