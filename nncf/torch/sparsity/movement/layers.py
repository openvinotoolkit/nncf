"""
 Copyright (c) 2019 Intel Corporation
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
from typing import List

import torch

from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.torch.functions import logit
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy

from torch.nn.modules import sparse
from torch import nn
import itertools as it
import numpy as np
from nncf.torch.sparsity.functions import apply_binary_mask as apply_binary_mask_impl
from nncf.torch.utils import is_tracing_state, no_jit_trace

class SparseStructure(str, Enum):
    FINE = "fine"
    BLOCK = "block"
    PER_DIM = "per_dim"

    @classmethod
    def from_str(cls, mode: str) -> 'SparseStructure':
        if mode == SparseStructure.FINE.value:
            return SparseStructure.FINE
        if mode == SparseStructure.BLOCK.value:
            return SparseStructure.BLOCK
        if mode == SparseStructure.PER_DIM.value:
            return SparseStructure.PER_DIM
        raise RuntimeError(f"Unknown sparse structure: {mode}."
                           f"List of supported: {[e.value for e in SparseStructure]}")


class SparseConfig:
    def __init__(self,
                 mode: SparseStructure,
                 sparse_factors: Optional[Tuple[int, int]] = None,
                 sparse_axis: Optional[int] = None):
        error_prefix = 'Invalid sparse config.'
        self.sparse_factors = None
        self.sparse_axis = None
        self.mode = mode
        if self.mode == SparseStructure.FINE:
            self.sparse_factors = (1, 1)

        if self.mode == SparseStructure.BLOCK:
            if sparse_factors is None:
                raise ValueError(
                    '{} Missing `sparse_factors`. Block sparsity structure expects it specified.'.format(error_prefix))
            if not (isinstance(sparse_factors, (tuple, list)) and len(sparse_factors) == 2):
                raise ValueError('{} Invalid format of `sparse_factors`. '
                                 'Block sparsity structure expects tuple of two numbers')
            self.sparse_factors = tuple(sparse_factors)

        if self.mode == SparseStructure.PER_DIM:
            if sparse_axis is None:
                raise ValueError(
                    '{} Missing `axis`. Sparsity structure per dimension expects it specified.'.format(error_prefix))
            self.sparse_axis = sparse_axis

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SparseConfig':
        """
        Creates the object from its config.
        """
        mode_str = config.get('mode', SparseStructure.FINE.value)
        mode = SparseStructure.from_str(mode_str)
        sparse_factors = config.get('sparse_factors')
        axis = config.get('axis')
        return cls(mode, sparse_factors, axis)

    def __str__(self):
        return f'{self.mode.value, self.sparse_factors}'


class SparseConfigByScope:
    def __init__(self, sparse_config: SparseConfig, target_scopes: str):
        self.sparse_config = sparse_config
        self.target_scopes = target_scopes

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SparseConfigByScope':
        """
        Creates the object from its representation .
        """
        error_prefix = 'Invalid sparse structure by scopes {}.'.format(config)
        target_scopes = config.get('target_scopes')
        if not target_scopes:
            raise ValueError('{} Missing `target_scopes`.'.format(error_prefix))
        sparse_config = SparseConfig.from_config(config)
        return cls(sparse_config, target_scopes)


@COMPRESSION_MODULES.register()
class MovementSparsifier(nn.Module):
    def __init__(self,
                 target_module_node,
                 sparse_cfg: SparseConfig,
                 frozen=True,
                 compression_lr_multiplier=None,
                 eps=1e-6):
        super().__init__()

        DEBUG=False

        self.target_module_node = target_module_node
        self.prune_bias = target_module_node.layer_attributes.bias

        self.frozen = frozen
        self.eps = eps
        self.lmbd = 0.5 # module_level_loss_weightage
        self.masking_threshold = -999 # This must be sufficiently small # TODO: there might be dependency

        weight_shape = target_module_node.layer_attributes.get_weight_shape()
        self.weight_ctx = BinaryMask(weight_shape)
        self.sparse_factors = self._get_sparse_factors(weight_shape, sparse_cfg)
        self.sparse_structure = sparse_cfg.mode
        self._weight_importance_shape, self._bool_expand_importance = self._get_importance_shape(
            weight_shape, self.sparse_factors, self.sparse_structure)
        self._weight_importance = CompressionParameter(
                                torch.rand(self._weight_importance_shape) if DEBUG is True else torch.zeros(self._weight_importance_shape),
                                requires_grad=not self.frozen,
                                compression_lr_multiplier=compression_lr_multiplier)
        self.weight_ctx.binary_mask = binary_mask_by_threshold(
                                            self._expand_importance(self._weight_importance), 
                                            self._masking_threshold
                                        )

        if self.prune_bias is True:
            bias_shape = target_module_node.layer_attributes.get_bias_shape()
            self.bias_ctx = BinaryMask(bias_shape)
            self._bias_importance_shape = self._weight_importance_shape[0]
            self._bias_importance = CompressionParameter(
                                torch.rand(self._bias_importance_shape) if DEBUG is True else torch.zeros(self._bias_importance_shape),
                                requires_grad=not self.frozen,
                                compression_lr_multiplier=compression_lr_multiplier)
            self.bias_ctx.binary_mask = binary_mask_by_threshold(
                                            self._expand_importance(self._bias_importance, isbias=True), 
                                            self._masking_threshold
                                        )

        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def importance(self):
        return self._weight_importance.data

    @property
    def masking_threshold(self):
        return self._masking_threshold
    
    @masking_threshold.setter
    def masking_threshold(self, threshold_value):
        self._masking_threshold = threshold_value

    @property
    def lmbd(self):
        return self._lmbd
    
    @lmbd.setter
    def lmbd(self, module_level_loss_weightage):
        self._lmbd = module_level_loss_weightage

    def freeze_importance(self):
        self.frozen = True
        self._weight_importance.requires_grad=False
        if self.prune_bias is True:
            self._bias_importance.requires_grad=False

    def unfreeze_importance(self):
        self.frozen = False
        self._weight_importance.requires_grad=True
        if self.prune_bias is True:
            self._bias_importance.requires_grad=True


    def extra_repr(self):
        return 'sparse_structure: {}'.format(self.sparse_structure.value, self.sparse_factors)

    def forward(self, weight, bias):
        if is_tracing_state():
            with no_jit_trace():
                return weight.mul_(self.weight_ctx.binary_mask), bias.mul_(self.bias_ctx.binary_mask)
        tmp_wtensor, tmp_btensor = self._calc_training_binary_mask(weight, bias)
        wtensor = apply_binary_mask_impl(tmp_wtensor, weight)
        btensor = apply_binary_mask_impl(tmp_btensor, bias)
        return wtensor, btensor

    def _calc_training_binary_mask(self, weight=None, bias=None): # TODO: unnecessary arguments
        if self.training and not self.frozen:
            w_mask = binary_mask_by_threshold(
                self._expand_importance(self._weight_importance), 
                self._masking_threshold
            )
            self.weight_ctx.binary_mask = w_mask
            
            b_mask = binary_mask_by_threshold(
                self._expand_importance(self._bias_importance, isbias=True), 
                self._masking_threshold
            )
            self.bias_ctx.binary_mask = b_mask
            return w_mask, b_mask
        else:
            return self.weight_ctx.binary_mask, self.bias_ctx.binary_mask


    def apply_binary_mask(self, param_tensor, isbias=False):
        if isbias is True:
            return self.bias_ctx.apply_binary_mask(param_tensor)
        return self.weight_ctx.apply_binary_mask(param_tensor)

    @staticmethod
    def _get_sparse_factors(weight_shape, sparse_config: SparseConfig):
        sparse_factors = sparse_config.sparse_factors
        if sparse_config.mode == SparseStructure.BLOCK:
            r, c = sparse_factors
            assert weight_shape[0] % r == 0, "r: {} is not a factor of dim axes 0".format(r)
            assert weight_shape[1] % c == 0, "c: {} is not a factor of dim axes 1".format(c)

        if sparse_config.mode == SparseStructure.PER_DIM:
            if sparse_config.sparse_axis < 0 or sparse_config.sparse_axis >= len(weight_shape):
                raise ValueError("Invalid axis id {}, axes range {}".format(
                    sparse_config.sparse_axis,
                    list(range(len(weight_shape)))))
            sparse_factors = deepcopy(weight_shape)
            sparse_factors[sparse_config.sparse_axis] = 1
            sparse_factors = tuple(sparse_factors)
        return sparse_factors

    @staticmethod
    def _get_importance_shape(weight_shape, sparse_factors: Tuple[int, int], sparse_structure: SparseStructure):
        # TODO:remove  weight_shape, r=32, c=32):
        if sparse_structure == SparseStructure.FINE:
            return weight_shape, False

        if sparse_structure == SparseStructure.BLOCK:
            r, c = sparse_factors
            return (weight_shape[0]//r, weight_shape[1]//c), True

        if sparse_structure == SparseStructure.PER_DIM:
            score_shape = []
            for axes, (dim, factor) in enumerate(zip(weight_shape, sparse_factors)):
                assert dim % factor == 0, "{} is not a factor of axes {} with dim size {}".format(factor, axes, dim)
                score_shape.append(dim//factor)
            return score_shape, True

    def _expand_importance(self, importance, isbias=False):
        #TODO only works dense layer for now
        if self._bool_expand_importance:
            if isbias is False:
                return importance.repeat_interleave(
                    self.sparse_factors[0], dim=0).repeat_interleave(
                    self.sparse_factors[1], dim=1)
            else:
                return importance.repeat_interleave(
                    self.sparse_factors[0], dim=0)
        return importance

    def loss(self):
        return self.lmbd * (
            torch.norm(torch.sigmoid(self._expand_importance(self._weight_importance)), p=1) / self._weight_importance.numel() + \
            torch.norm(torch.sigmoid(self._expand_importance(self._bias_importance, isbias=True)), p=1) / self._bias_importance.numel()
        )

    def get_structured_mask(self, grain_size=None):
        if grain_size is None:
            grain_size = self.sparse_factors
        structured_mask_shape = [dim//grain_size[axes] for axes, dim in enumerate(list(self.weight_ctx.binary_mask.shape))]
        temp_shape = list(it.chain(*zip(list(structured_mask_shape), list(grain_size))))
        structured_mask = self.weight_ctx.binary_mask.detach().clone()
        structured_mask = structured_mask.reshape(temp_shape)
        structured_mask = structured_mask.amax(dim=(tuple((np.arange(len(self.weight_ctx.binary_mask.shape)) * 2 + 1))))
        # print("Mask Shape from {} to {}".format(structured_mask.shape, self.weight_ctx.binary_mask.shape))
        if self.prune_bias is True:
            structured_bias_mask_shape = structured_mask_shape[0]
            structured_bias_mask = self.bias_ctx.binary_mask.detach().clone()
            structured_bias_mask = structured_bias_mask.reshape((structured_bias_mask_shape, -1))
            structured_bias_mask = structured_bias_mask.amax(dim=1)
            dim_aligned = structured_bias_mask.repeat(structured_mask.shape[1]).reshape(-1, structured_mask.shape[1])
            structured_mask = structured_mask.logical_or(dim_aligned).to(torch.float32)
        return structured_mask

    def set_structured_mask(self, structured_mask):
        self.weight_ctx.binary_mask=structured_mask
        if self.prune_bias is True:
            self.bias_ctx.binary_mask=structured_mask.amax(dim=1)

class MaskCalculationHook():
    def __init__(self, module):
        # pylint: disable=protected-access
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        # module.weight_ctx.binary_mask = binary_mask_by_threshold(
        #                         module._expand_importance(module._weight_importance), 
        #                         module.masking_threshold
        #                      )
        destination[prefix + 'weight_ctx._binary_mask'] = module.weight_ctx.binary_mask

        if module.prune_bias is True:
            # module.bias_ctx.binary_mask = binary_mask_by_threshold(
            #                     module._expand_importance(module._bias_importance, isbias=True), 
            #                     module.masking_threshold
            #                 )
            destination[prefix + 'bias_ctx._binary_mask'] = module.bias_ctx.binary_mask
        return destination

    def close(self):
        self.hook.remove()